use rust_regex_compiler::RegexToFSTBuilder;

#[test]
fn huge_range() {
    let valid_char = "([a-zA-Z0-9]|(_)|(-))";
    let regex = format!("([A-Za-z0-9]({valid_char}{{0,5}}[A-Za-z0-9])?)", valid_char=valid_char)
        .to_string();
    let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
}