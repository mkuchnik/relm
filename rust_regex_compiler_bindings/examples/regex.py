import rust_regex_compiler_bindings

regex = "a|b|(cdef)*"
re_repr = rust_regex_compiler_bindings.regex_to_fst(regex)
print(re_repr)