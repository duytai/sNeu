use std::io::{self, Read};
use std::{panic, process};
// use serde_json::{Result, Value};
use std::str;

fn do_it(data: Vec<u8>) {
    if data.len() < 6 {return}
    if data[0] != b'q' {return}
    if data[1] != b'w' {return}
    if data[2] != b'e' {return}
    if data[3] != b'r' {return}
    if data[4] != b't' {return}
    if data[5] != b'y' {return}
    panic!("BOOM")
    // let sparkle_heart = str::from_utf8(&data).unwrap();
    // let v: Value = serde_json::from_str(sparkle_heart).unwrap();
}

fn main() {
    let mut input = vec![];
    let result = io::stdin().read_to_end(&mut input);
    if result.is_err() {
        return;
    }
    let was_panic = panic::catch_unwind(|| {
        do_it(input);
    });
    if was_panic.is_err() {
        process::abort();
    }
}
