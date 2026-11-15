use super::{input::Expected, Arguments, Result};
use serde::Serialize;
use serde_json::Value;
use std::{path::Path, process::Stdio, time::Duration};
use tokio::process::Command;

#[derive(Debug, Serialize)]
pub struct Media {
    pub width: u64,
    pub height: u64,
    pub frames: u64,
    pub fps: f64,
    pub audio: bool,
}

async fn output(mut command: Command, seconds: u64) -> Result<std::process::Output> {
    command.stdin(Stdio::null()).kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(seconds), command.output()).await??;
    if !output.status.success() {
        return Err(format!(
            "media tool {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output)
}

pub async fn check_tools(arguments: &Arguments) -> Result<()> {
    for program in [&arguments.ffprobe, &arguments.ffmpeg] {
        let mut command = Command::new(program);
        command.arg("-version");
        output(command, arguments.validation_timeout_seconds).await?;
    }
    Ok(())
}

fn numeric(value: &Value, key: &str) -> Result<f64> {
    let value = value.get(key).ok_or_else(|| format!("missing {key}"))?;
    if let Some(number) = value.as_f64() {
        return Ok(number);
    }
    Ok(value
        .as_str()
        .ok_or_else(|| format!("invalid {key}"))?
        .parse()?)
}

fn media(probe: &Value, expected: &Expected, require_audio: bool) -> Result<Media> {
    let streams = probe
        .get("streams")
        .and_then(Value::as_array)
        .ok_or("ffprobe returned no streams")?;
    let video = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(Value::as_str) == Some("video"))
        .ok_or("no video stream")?;
    let frame_rate = video
        .get("avg_frame_rate")
        .and_then(Value::as_str)
        .ok_or("missing average frame rate")?;
    let (numerator, denominator) = frame_rate.split_once('/').ok_or("invalid frame rate")?;
    let fps = numerator.parse::<f64>()? / denominator.parse::<f64>()?;
    let width = numeric(video, "width")?;
    let height = numeric(video, "height")?;
    let frames = numeric(video, "nb_read_frames")?;
    if width != expected.width as f64
        || height != expected.height as f64
        || frames != expected.frames as f64
        || !fps.is_finite()
        || (fps - expected.fps).abs() > expected.fps * 0.0001
    {
        return Err(format!("video shape {width}x{height}/{frames} frames/{fps} fps differs from requested {expected:?}").into());
    }
    let audio = streams.iter().any(|s| {
        s.get("codec_type").and_then(Value::as_str) == Some("audio")
            && numeric(s, "sample_rate").is_ok_and(|rate| rate.is_finite() && rate > 0.0)
            && numeric(s, "channels").is_ok_and(|channels| channels >= 1.0)
            && numeric(s, "nb_read_frames").is_ok_and(|frames| frames >= 1.0)
    });
    if require_audio && !audio {
        return Err("missing decodable nonempty audio stream".into());
    }
    Ok(Media {
        width: expected.width,
        height: expected.height,
        frames: expected.frames,
        fps,
        audio,
    })
}

pub async fn validate(arguments: &Arguments, path: &Path, expected: &Expected) -> Result<Media> {
    let mut command = Command::new(&arguments.ffprobe);
    command
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
        ])
        .arg(path);
    let probe = output(command, arguments.validation_timeout_seconds).await?;
    tokio::fs::write(path.with_extension("ffprobe.json"), &probe.stdout).await?;
    let media = media(
        &serde_json::from_slice(&probe.stdout)?,
        expected,
        arguments.require_audio,
    )?;
    let mut command = Command::new(&arguments.ffmpeg);
    command
        .args(["-nostdin", "-v", "error", "-xerror", "-i"])
        .arg(path)
        .args(["-map", "0:v:0", "-map", "0:a?", "-f", "null", "-"]);
    let decoded = output(command, arguments.validation_timeout_seconds).await?;
    tokio::fs::write(path.with_extension("decode.stderr"), decoded.stderr).await?;
    Ok(media)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    fn probe(frames: &str) -> Value {
        serde_json::json!({"streams":[{"codec_type":"video", "width":832, "height":480, "nb_read_frames":frames, "avg_frame_rate":"24/1"}]})
    }
    #[test]
    fn rejects_truncated_video_and_absent_required_audio() {
        let expected = Expected {
            width: 832,
            height: 480,
            frames: 124,
            fps: 24.0,
        };
        assert!(media(&probe("124"), &expected, false).is_ok());
        assert!(media(&probe("120"), &expected, false).is_err());
        assert!(media(&probe("124"), &expected, true).is_err());
    }

    #[tokio::test]
    #[ignore = "requires ffmpeg and ffprobe executables"]
    async fn validates_a_complete_audio_video_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fixture.mp4");
        let mut command = Command::new("ffmpeg");
        command
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=16x16:r=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-t",
                "0.125",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
            ])
            .arg(&path);
        output(command, 30).await.unwrap();
        let arguments = Arguments::try_parse_from([
            "test",
            "--base-url",
            "http://localhost:1",
            "--request-json",
            "unused",
            "--provenance",
            "unused",
            "--server-config",
            "unused",
            "--output",
            "unused",
            "--require-audio",
        ])
        .unwrap();
        let expected = Expected {
            width: 16,
            height: 16,
            frames: 3,
            fps: 24.0,
        };
        let artifact = validate(&arguments, &path, &expected).await.unwrap();
        assert!(artifact.audio);
        assert_eq!(artifact.frames, 3);
    }
}
