mod util;
mod analyze;

const PRECISION: usize = 100;
const MATCH_THRESHOLD: f32 = 32.0;

const BLOCK_UPDATE: usize = 30;
const STREAM_JUMP: usize = 30;

const VID_SCALE: f64 = 0.3;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();

    let f = match args.get(1) {
        Some(strategy) if strategy == "streaming" => streaming,
        Some(strategy) if strategy == "blocking" => blocking,
        _ => panic!("invalid strategy"),
    };

    let mut scan = util::capture("data/scan.mp4").unwrap();
    analyze::orb_features(&mut scan, PRECISION).unwrap();

    let mut env = util::capture("data/env.mp4").unwrap();
    analyze::orb_features(&mut env, PRECISION).unwrap();

    f(scan, env);
}

fn blocking(scan: Vec<util::Frame>, env: Vec<util::Frame>) {
    println!("Begin Blocking...");
    let mut best_scan_i = 0;
    let mut env_i = 0;

    for env_frame in env.iter() {
        if env_i % BLOCK_UPDATE == 0 {
            best_scan_i = analyze::most_similar(&scan, &env[env_i], MATCH_THRESHOLD).unwrap();
        }

        util::show(&scan[best_scan_i], false, VID_SCALE, "scan.mp4").unwrap();
        util::show(env_frame, true, VID_SCALE, "env.mp4").unwrap();
        if util::key_pressed('q', false).unwrap() {
            break;
        }
        env_i += 1;
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::sync::Arc;

fn streaming(_scan: Vec<util::Frame>, _env: Vec<util::Frame>) {
    println!("Begin Streaming...");
    let scan = Arc::new(_scan);
    let env = Arc::new(_env);

    let best_scan_i = Arc::new(AtomicUsize::new(0));
    let env_i = Arc::new(AtomicUsize::new(0));

    // Thread Data
    let h_env = Arc::clone(&env);
    let h_scan = Arc::clone(&scan);
    let h_best_scan_i = Arc::clone(&best_scan_i);
    let h_env_i = Arc::clone(&env_i);

    let handle = thread::spawn(move || {
        let mut env_i = h_env_i.load(Ordering::SeqCst);
        let mut best_i; 

        while env_i+STREAM_JUMP < h_env.len() {
            best_i = analyze::most_similar(&h_scan, &h_env[env_i+STREAM_JUMP], 0.9).unwrap();
            h_best_scan_i.store(best_i, Ordering::SeqCst);

            env_i = h_env_i.load(Ordering::SeqCst);
        }
    });

    for env_frame in env.iter() {
        let local_best_scan_i = best_scan_i.load(Ordering::SeqCst);
        util::show(&scan[local_best_scan_i], false, VID_SCALE, "scan.mp4").unwrap();
        util::show(env_frame, true, VID_SCALE, "env.mp4").unwrap();

        if util::key_pressed('q', false).unwrap() {
            break;
        }

        env_i.fetch_add(1, Ordering::SeqCst);
    }
    handle.join().unwrap();
}
