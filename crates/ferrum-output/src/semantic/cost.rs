use ferrum_types::{FerrumError, Result};
use serde::ser::{self, Serialize};
use std::fmt::{self, Write};

/// Logical payload size, without constructing a JSON tree or encoded buffer.
/// Strings use UTF-8 bytes; numbers use their Serialize primitive width. Dynamic
/// map keys count, static struct field names do not. Containers, options and
/// unit values each count one byte so empty nested data cannot be free.
///
/// This is not an allocator/RSS bound. Use incremental append lengths in token
/// loops and call this only when retaining a new DTO or freezing its ownership.
pub fn semantic_cost<T: Serialize + ?Sized>(value: &T) -> Result<usize> {
    value.serialize(Counter).map_err(|error| {
        FerrumError::internal(format!("cannot count retained output content: {error}"))
    })
}

#[derive(Debug)]
struct CostError(String);
impl fmt::Display for CostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for CostError {}
impl ser::Error for CostError {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}
type CostResult<T> = std::result::Result<T, CostError>;

fn add(left: usize, right: usize) -> CostResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| CostError("logical content size overflow".into()))
}

struct Counter;
struct Compound(usize);
impl Compound {
    fn field<T: Serialize + ?Sized>(&mut self, value: &T) -> CostResult<()> {
        self.0 = add(self.0, value.serialize(Counter)?)?;
        Ok(())
    }
}

macro_rules! primitive {
    ($method:ident, $ty:ty, $bytes:expr) => {
        fn $method(self, _: $ty) -> CostResult<usize> {
            Ok($bytes)
        }
    };
}

impl ser::Serializer for Counter {
    type Ok = usize;
    type Error = CostError;
    type SerializeSeq = Compound;
    type SerializeTuple = Compound;
    type SerializeTupleStruct = Compound;
    type SerializeTupleVariant = Compound;
    type SerializeMap = Compound;
    type SerializeStruct = Compound;
    type SerializeStructVariant = Compound;

    primitive!(serialize_bool, bool, 1);
    primitive!(serialize_i8, i8, 1);
    primitive!(serialize_i16, i16, 2);
    primitive!(serialize_i32, i32, 4);
    primitive!(serialize_i64, i64, 8);
    primitive!(serialize_i128, i128, 16);
    primitive!(serialize_u8, u8, 1);
    primitive!(serialize_u16, u16, 2);
    primitive!(serialize_u32, u32, 4);
    primitive!(serialize_u64, u64, 8);
    primitive!(serialize_u128, u128, 16);
    primitive!(serialize_f32, f32, 4);
    primitive!(serialize_f64, f64, 8);
    primitive!(serialize_char, char, 4);

    fn serialize_str(self, value: &str) -> CostResult<usize> {
        Ok(value.len())
    }
    fn serialize_bytes(self, value: &[u8]) -> CostResult<usize> {
        add(1, value.len())
    }
    fn serialize_none(self) -> CostResult<usize> {
        Ok(1)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> CostResult<usize> {
        add(1, value.serialize(self)?)
    }
    fn serialize_unit(self) -> CostResult<usize> {
        Ok(1)
    }
    fn serialize_unit_struct(self, _: &'static str) -> CostResult<usize> {
        Ok(1)
    }
    fn serialize_unit_variant(self, _: &'static str, _: u32, _: &'static str) -> CostResult<usize> {
        Ok(1)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> CostResult<usize> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        value: &T,
    ) -> CostResult<usize> {
        add(1, value.serialize(self)?)
    }
    fn serialize_seq(self, _: Option<usize>) -> CostResult<Compound> {
        Ok(Compound(1))
    }
    fn serialize_tuple(self, _: usize) -> CostResult<Compound> {
        Ok(Compound(1))
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> CostResult<Compound> {
        Ok(Compound(1))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> CostResult<Compound> {
        Ok(Compound(1))
    }
    fn serialize_map(self, _: Option<usize>) -> CostResult<Compound> {
        Ok(Compound(1))
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> CostResult<Compound> {
        Ok(Compound(1))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> CostResult<Compound> {
        Ok(Compound(1))
    }
    fn collect_str<T: fmt::Display + ?Sized>(self, value: &T) -> CostResult<usize> {
        struct DisplayCounter(usize);
        impl fmt::Write for DisplayCounter {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                self.0 = self.0.checked_add(text.len()).ok_or(fmt::Error)?;
                Ok(())
            }
        }
        let mut output = DisplayCounter(0);
        write!(&mut output, "{value}")
            .map_err(|_| CostError("display content size overflow".into()))?;
        Ok(output.0)
    }
}

macro_rules! compound {
    ($trait:ident, $method:ident) => {
        impl ser::$trait for Compound {
            type Ok = usize;
            type Error = CostError;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> CostResult<()> {
                self.field(value)
            }
            fn end(self) -> CostResult<usize> {
                Ok(self.0)
            }
        }
    };
}
compound!(SerializeSeq, serialize_element);
compound!(SerializeTuple, serialize_element);
compound!(SerializeTupleStruct, serialize_field);
compound!(SerializeTupleVariant, serialize_field);

impl ser::SerializeMap for Compound {
    type Ok = usize;
    type Error = CostError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> CostResult<()> {
        self.field(key)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> CostResult<()> {
        self.field(value)
    }
    fn end(self) -> CostResult<usize> {
        Ok(self.0)
    }
}

macro_rules! structure {
    ($trait:ident) => {
        impl ser::$trait for Compound {
            type Ok = usize;
            type Error = CostError;
            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                _: &'static str,
                value: &T,
            ) -> CostResult<()> {
                self.field(value)
            }
            fn end(self) -> CostResult<usize> {
                Ok(self.0)
            }
        }
    };
}
structure!(SerializeStruct);
structure!(SerializeStructVariant);
