//! HTTP payload ownership for bounded inference output.
//!
//! Producers reserve before allocating and serializing a frame. This final
//! boundary verifies that its backing storage fits the existing reservation;
//! it cannot retroactively authorize an allocation. Clones and slices of the
//! returned `Bytes` retain that reservation until the backing owner is dropped.
//! This observes local transport ownership, not receipt by the remote client.

use axum::body::{Body, Bytes};
use ferrum_interfaces::output_credit::LeasedOutput;
use futures::{Stream, StreamExt};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CreditedWireError {
    #[error("an output frame requires reserved event capacity")]
    MissingEventCredit,
    #[error("output backing storage uses {capacity} bytes but owns only {reserved}")]
    InsufficientByteCredit { capacity: usize, reserved: usize },
}

struct WireOwner(LeasedOutput<Vec<u8>>);

impl AsRef<[u8]> for WireOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.payload().as_slice()
    }
}

/// Transfer a serialized frame without copying or detaching its credit.
///
/// An uncharged conversion to `Vec`/`BytesMut`, compression buffer, or other
/// copied representation is outside this contract. Such transforms must reserve
/// their own simultaneous storage before copying, and preserve ownership too.
pub fn into_bytes(frame: LeasedOutput<Vec<u8>>) -> Result<Bytes, CreditedWireError> {
    if frame.credit().events == 0 {
        return Err(CreditedWireError::MissingEventCredit);
    }
    let capacity = frame.payload().capacity();
    let reserved = frame.credit().bytes;
    if capacity > reserved {
        return Err(CreditedWireError::InsufficientByteCredit { capacity, reserved });
    }
    Ok(Bytes::from_owner(WireOwner(frame)))
}

/// A complete JSON or other single-frame response retains its reservation in
/// the body and in any data frames subsequently handed to the HTTP transport.
pub fn single_body(frame: LeasedOutput<Vec<u8>>) -> Result<Body, CreditedWireError> {
    into_bytes(frame).map(Body::from)
}

/// Stream already serialized, credited frames. The producer owns framing and
/// protocol order; this layer neither synthesizes uncharged error events nor
/// introduces another queue. Dropping the body drops unpolled owned frames.
pub fn stream_body<S>(frames: S) -> Body
where
    S: Stream<Item = LeasedOutput<Vec<u8>>> + Send + 'static,
{
    Body::from_stream(frames.map(into_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::output_credit::{
        OutputAccountLimits, OutputCreditAccount, OutputCreditAmount, OutputCreditAttempt,
        OutputCreditLane, OutputCreditPool, OutputPoolLimits,
    };
    use ferrum_types::RequestId;
    use futures::{stream, FutureExt};

    fn account() -> (OutputCreditPool, OutputCreditAccount) {
        let maximum = OutputCreditAmount {
            events: 4,
            bytes: 512,
            projection_bytes: 256,
        };
        let pool = OutputCreditPool::new(OutputPoolLimits {
            maximum,
            max_total_bytes: 768,
            max_open_accounts: 1,
        })
        .unwrap();
        let account = pool
            .open_request(
                RequestId::new(),
                OutputAccountLimits {
                    maximum,
                    terminal: OutputCreditAmount {
                        events: 1,
                        bytes: 32,
                        projection_bytes: 0,
                    },
                },
            )
            .unwrap();
        (pool, account)
    }

    fn reserve_frame(
        account: &OutputCreditAccount,
        payload: Vec<u8>,
        bytes: usize,
        events: usize,
    ) -> LeasedOutput<Vec<u8>> {
        let amount = OutputCreditAmount {
            events,
            bytes,
            projection_bytes: 0,
        };
        match account.try_reserve(OutputCreditLane::Data, amount).unwrap() {
            OutputCreditAttempt::Reserved(reservation) => reservation.into_output(payload),
            OutputCreditAttempt::Full(_) => panic!("fixture requires available output capacity"),
        }
    }

    fn frame(account: &OutputCreditAccount, text: &[u8]) -> LeasedOutput<Vec<u8>> {
        let payload = text.to_vec();
        let capacity = payload.capacity();
        reserve_frame(account, payload, capacity, 1)
    }

    #[test]
    fn clones_and_slices_keep_closed_account_storage_until_last_owner() {
        let (pool, account) = account();
        let frame = frame(&account, b"data: hello\n\n");
        let amount = frame.credit();
        let original_pointer = frame.payload().as_ptr();
        let bytes = into_bytes(frame).unwrap();
        assert_eq!(original_pointer, bytes.as_ptr());
        let cloned = bytes.clone();
        let slice = cloned.slice(6..11);
        account.close();
        assert_eq!(pool.snapshot().retained_accounts, 1);
        let mut wake = pool.subscribe();

        drop(bytes);
        drop(cloned);
        assert_eq!(slice.as_ref(), b"hello");
        assert_eq!(pool.snapshot().data_used, amount);
        assert!(wake.changed().now_or_never().is_none());

        drop(slice);
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
        assert_eq!(pool.snapshot().retained_accounts, 0);
        assert!(wake.changed().now_or_never().unwrap().is_ok());
    }

    #[tokio::test]
    async fn dropping_body_releases_unpolled_frames_but_not_returned_bytes() {
        let (pool, account) = account();
        let first = frame(&account, b"data: first\n\n");
        let first_credit = first.credit();
        let second = frame(&account, b"data: second\n\n");
        let mut body = stream_body(stream::iter([first, second])).into_data_stream();
        let bytes = body.next().await.unwrap().unwrap();
        let transport_clone = bytes.clone();
        drop(bytes);
        drop(body);
        assert_eq!(pool.snapshot().data_used, first_credit);
        assert_eq!(transport_clone.as_ref(), b"data: first\n\n");
        drop(transport_clone);
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    }

    #[tokio::test]
    async fn single_response_body_preserves_credit_after_body_is_consumed() {
        let (pool, account) = account();
        let frame = frame(&account, b"{\"text\":\"hello\"}");
        let credit = frame.credit();
        let mut body = single_body(frame).unwrap().into_data_stream();
        let bytes = body.next().await.unwrap().unwrap();
        assert!(body.next().await.is_none());
        drop(body);
        assert_eq!(pool.snapshot().data_used, credit);
        drop(bytes);
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    }

    #[test]
    fn dropping_unpolled_body_closes_producer_and_releases_queued_frames() {
        let (pool, account) = account();
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        assert!(sender
            .try_send(frame(&account, b"data: queued\n\n"))
            .is_ok());
        let body = stream_body(tokio_stream::wrappers::ReceiverStream::new(receiver));
        let mut wake = pool.subscribe();
        assert_eq!(pool.snapshot().data_used.events, 1);
        drop(body);
        assert!(sender.is_closed());
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
        assert!(wake.changed().now_or_never().unwrap().is_ok());
    }

    #[test]
    fn byte_credit_covers_backing_capacity_and_each_frame_requires_event_credit() {
        let (pool, account) = account();
        let mut spare_capacity = Vec::with_capacity(64);
        spare_capacity.push(b'x');
        let capacity = spare_capacity.capacity();
        let undersized = reserve_frame(&account, spare_capacity, 1, 1);
        assert_eq!(
            into_bytes(undersized).unwrap_err(),
            CreditedWireError::InsufficientByteCredit {
                capacity,
                reserved: 1
            }
        );
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);

        let no_event = reserve_frame(&account, b"x".to_vec(), 1, 0);
        assert_eq!(
            into_bytes(no_event).unwrap_err(),
            CreditedWireError::MissingEventCredit
        );
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    }

    #[test]
    fn empty_body_storage_is_still_retained_until_drop() {
        let (pool, account) = account();
        let payload = Vec::with_capacity(16);
        let capacity = payload.capacity();
        let frame = reserve_frame(&account, payload, capacity, 1);
        let credit = frame.credit();
        let bytes = into_bytes(frame).unwrap();
        assert!(bytes.is_empty());
        assert_eq!(pool.snapshot().data_used, credit);
        drop(bytes);
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    }

    #[tokio::test]
    async fn invalid_stream_frame_reports_transport_error_and_releases_ownership() {
        let (pool, account) = account();
        let bad = reserve_frame(&account, b"x".to_vec(), 1, 0);
        let mut body = stream_body(stream::iter([bad])).into_data_stream();
        assert!(body.next().await.unwrap().is_err());
        drop(body);
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    }
}
