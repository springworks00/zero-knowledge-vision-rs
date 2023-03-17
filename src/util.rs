use opencv::{
    core::{self, Scalar, KeyPoint, Point, Mat},
    highgui,
    imgproc,
    videoio,
};
use ndarray::Array2;
use std::path::Path;
use std::convert::TryInto;
use std::io::Write;

use opencv::core::Vector;
use opencv::core::KeyPointTraitConst;
use opencv::prelude::MatTraitConst;
use opencv::prelude::VideoCaptureTrait;
use opencv::prelude::VideoCaptureTraitConst;
use opencv::features2d::draw_keypoints;

pub struct Frame {
    pub raw: Mat,
    pub kps: Option<Vector<KeyPoint>>,
    pub dcs: Option<Mat>,
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl Frame {
    pub fn new(raw: Mat) -> Self {
        Self {
            raw: raw,
            kps: None,
            dcs: None,
        }
    }
}

pub fn status(msg: &str) {
    print!("\r{}", msg);
    std::io::stdout().flush().unwrap();
}

pub fn capture(path: &str) -> Result<Vec<Frame>, opencv::Error> {
    let mut cap = videoio::VideoCapture::from_file(path, videoio::CAP_ANY)?;

    if !cap.is_opened()? {
        return Err(opencv::Error::new(0, "File not found".to_string()));
    }

    let cap_len: i32 = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as i32;
    let mut frames = Vec::new();
    let mut i: i32 = 1;

    while cap.is_opened()? {
        let mut raw = Mat::default();
        let ok = cap.read(&mut raw)?;

        if !ok {
            break;
        }

        frames.push(Frame::new(raw));
        status(&format!("({}) reading frames: {}%", path, 100 * i / cap_len));
        i += 1;
    }

    println!();
    Ok(frames)
}

pub fn error_frame(shape: core::Size) -> Result<Frame, opencv::Error> {
    let mut black = Mat::new_rows_cols_with_default(shape.height, shape.width, core::CV_8UC3, Scalar::all(0.0))?;

    imgproc::put_text(
        &mut black,
        "OBJECT NOT FOUND",
        Point::new(100, 100),
        imgproc::FONT_HERSHEY_SIMPLEX,
        2.0,
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        3,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(Frame::new(black))
}

pub fn vec_to_vector_keypoint(keypoints: &Vec<KeyPoint>) -> Vector<KeyPoint> {
    let keypoints_ref_iter = keypoints.iter().map(|kp| {
        let kp = unsafe { &*(kp as *const KeyPoint) };
        KeyPoint::new_coords(
            kp.pt().x,
            kp.pt().y,
            kp.size(),
            kp.angle(),
            kp.response(),
            kp.octave(),
            kp.class_id(),
        ).unwrap()
    });

    Vector::from_iter(keypoints_ref_iter)
}


pub fn show(frame: &Frame, kps: bool, scale: f64, title: &str) -> Result<(), opencv::Error> {
    let mut raw = frame.raw.clone();

    if kps {
        draw_keypoints(
            &frame.raw.clone(),
            frame.kps.as_ref().unwrap(),
            //&vec_to_vector_keypoint(frame.kps.as_ref().unwrap()),
            &mut raw,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            opencv::features2d::DrawMatchesFlags::DRAW_RICH_KEYPOINTS,
        )?;
    }

    let width = (raw.cols() as f64 * scale).round() as i32;
    let height = (raw.rows() as f64 * scale).round() as i32;
    let mut resized_raw = Mat::default();

    imgproc::resize(
        &raw,
        //&raw,
        &mut resized_raw,
        core::Size::new(width, height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    highgui::imshow(title, &resized_raw)?;
    Ok(())
}

pub fn destroy(title: &str) -> Result<(), opencv::Error> {
    highgui::destroy_window(title)?;
    Ok(())
}

pub fn key_pressed(key: char, wait: bool) -> Result<bool, opencv::Error> {
    let wait_time = if wait { -1 } else { 1 };
    let pressed_key = highgui::wait_key(wait_time)? as u8 as char;
    Ok(pressed_key == key)
}
