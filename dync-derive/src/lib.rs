use std::collections::{HashMap, HashSet};

use proc_macro2::{Span, TokenStream};
use quote::{quote, TokenStreamExt};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::*;

type GenericsMap = HashMap<Ident, Punctuated<TypeParamBound, Token![+]>>;
type ImplMap = HashMap<Path, (Path, Vec<(Path, Ident)>)>;

#[derive(Debug)]
struct Config {
    dync_crate_name: String,
    suffix: String,
    build_vtable_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            dync_crate_name: String::from("dync"),
            suffix: String::from("VTable"),
            build_vtable_only: false,
        }
    }
}

#[derive(Debug)]
struct DynAttrib {
    ident: Ident,
    eq: Option<Token![=]>,
    value: Option<Lit>,
}

impl Parse for DynAttrib {
    fn parse(input: ParseStream) -> Result<Self> {
        let ident = input.parse()?;
        let eq = input.parse()?;
        let value = input.parse()?;
        Ok(DynAttrib { ident, eq, value })
    }
}

impl Parse for Config {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut config = Config::default();
        let attribs: Punctuated<DynAttrib, Token![,]> = Punctuated::parse_terminated(input)?;
        for attrib in attribs.iter() {
            let name = attrib.ident.to_string();
            match (name.as_str(), &attrib.value) {
                ("build_vtable_only", None) => config.build_vtable_only = true,
                ("dync_crate_name", Some(Lit::Str(ref lit))) => {
                    config.dync_crate_name = lit.value().clone()
                }
                ("suffix", Some(Lit::Str(ref lit))) => config.suffix = lit.value().clone(),
                _ => {}
            }
        }
        Ok(config)
    }
}

/// A table of predefined VTables for std traits
fn builtin_impls(crate_name: &str) -> ImplMap {
    let crate_name = Ident::new(crate_name, Span::call_site());
    vec![
        (
            parse_quote! { Clone },
            vec![
                ("CloneFn", "clone_fn"),
                ("CloneFromFn", "clone_from_fn"),
                ("CloneIntoRawFn", "clone_into_raw_fn"),
            ],
        ),
        (parse_quote! { PartialEq }, vec![("EqFn", "eq_fn")]),
        (parse_quote! { Eq }, vec![]),
        (
            parse_quote! { std::hash::Hash },
            vec![("HashFn", "hash_fn")],
        ),
        (parse_quote! { std::fmt::Debug }, vec![("FmtFn", "fmt_fn")]),
    ]
    .into_iter()
    .map(|(trait_path, fns): (Path, Vec<(&str, &str)>)| {
        let fns = fns
            .into_iter()
            .map(|(fn_ty, fn_name)| {
                let fn_ty = Ident::new(fn_ty, Span::call_site());
                let fn_name = Ident::new(fn_name, Span::call_site());
                (parse_quote! { #fn_ty }, fn_name)
            })
            .collect();

        let seg = trait_path.segments.last().unwrap();
        let trait_name = &seg.ident;
        let vtable_name = Ident::new(&format!("{}VTable", &trait_name), Span::call_site());
        (
            trait_path,
            (parse_quote! { #crate_name::traits::#vtable_name }, fns),
        )
    })
    .collect()
}

fn builtin_trait_inheritance() -> HashMap<Path, HashSet<Path>> {
    let mut trait_inheritance: HashMap<Path, HashSet<Path>> = HashMap::new();
    let mut eq = HashSet::new();
    eq.insert(parse_quote! { PartialEq });
    trait_inheritance.insert(parse_quote! { Eq }, eq);
    trait_inheritance
}

#[proc_macro_attribute]
pub fn dync_mod(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let config: Config = syn::parse(attr).expect("Failed to parse attributes");

    let mut item_mod: ItemMod =
        syn::parse(item).expect("the dync_mod attribute applies only to mod definitions");

    validate_item_mod(&item_mod);

    let mut impls = builtin_impls(&config.dync_crate_name);

    let mut trait_inheritance = builtin_trait_inheritance();

    fill_inheritance_from_mod(&item_mod, &mut trait_inheritance);

    fill_inherited_impls(&mut impls, &config, &trait_inheritance);

    let mut dync_items = Vec::new();

    for item in item_mod.content.as_ref().unwrap().1.iter() {
        match item {
            Item::Trait(item_trait) => {
                dync_items.append(&mut construct_dync_items(
                    &item_trait,
                    &config,
                    &impls,
                    &trait_inheritance,
                ));
            }
            _ => {}
        }
    }

    item_mod.content.as_mut().unwrap().1.append(&mut dync_items);

    let tokens = quote! { #item_mod };

    tokens.into()
}

// Reject unsupported item traits.
fn validate_item_mod(item_mod: &ItemMod) {
    assert!(
        item_mod.content.is_some() && item_mod.semi.is_none(),
        "dync_mod attribute only works on modules containing trait definitions"
    );
}

fn fill_inherited_impls(
    impls: &mut ImplMap,
    config: &Config,
    trait_inheritance: &HashMap<Path, HashSet<Path>>,
) {
    for (trait_path, _) in trait_inheritance.iter() {
        let seg = trait_path.segments.last().unwrap();
        let trait_name = &seg.ident;
        let vtable_name = Ident::new(
            &format!("{}{}", &trait_name, &config.suffix),
            Span::call_site(),
        );
        // TODO: Construct the trait functions.
        impls
            .entry(trait_path.clone())
            .or_insert_with(|| (parse_quote! { #vtable_name }, vec![]));
    }
}

fn fill_inheritance_from_mod(
    item_mod: &ItemMod,
    trait_inheritance: &mut HashMap<Path, HashSet<Path>>,
) {
    // First we fill out all traits we know with their supertraits.
    for item in item_mod.content.as_ref().unwrap().1.iter() {
        match item {
            Item::Trait(item_trait) => {
                fill_inheritance_from_item_trait(&item_trait, trait_inheritance);
            }
            _ => {}
        }
    }

    // Then we eliminate propagate inherited traits down to the base traits.
    for item in item_mod.content.as_ref().unwrap().1.iter() {
        match item {
            Item::Trait(item_trait) => {
                flatten_inheritance(&item_trait, trait_inheritance);
            }
            _ => {}
        }
    }
}

fn fill_inheritance_from_item_trait(
    item_trait: &ItemTrait,
    trait_inheritance: &mut HashMap<Path, HashSet<Path>>,
) {
    let trait_name = item_trait.ident.clone();
    trait_inheritance
        .entry(parse_quote! { #trait_name })
        .or_insert_with(|| {
            item_trait
                .supertraits
                .iter()
                .filter_map(|bound| {
                    if let TypeParamBound::Trait(bound) = bound {
                        if bound.lifetimes.is_some() || bound.modifier != TraitBoundModifier::None {
                            // We are looking for recognizable traits only
                            None
                        } else {
                            // Generic traits are ignored
                            if bound
                                .path
                                .segments
                                .iter()
                                .all(|seg| seg.arguments.is_empty())
                            {
                                Some(bound.path.clone())
                            } else {
                                None
                            }
                        }
                    } else {
                        None
                    }
                })
                .collect()
        });
}

fn flatten_inheritance(
    item_trait: &ItemTrait,
    trait_inheritance: &mut HashMap<Path, HashSet<Path>>,
) {
    let trait_name = item_trait.ident.clone();
    let trait_path: Path = parse_quote! { #trait_name };
    let supers = trait_inheritance.remove(&trait_path).unwrap();
    let supers = supers
        .into_iter()
        .flat_map(|super_trait| {
            union_children(&super_trait, trait_inheritance)
                .into_iter()
                .chain(std::iter::once(super_trait.clone()))
        })
        .collect();
    assert!(trait_inheritance.insert(trait_path, supers).is_none());
}

fn union_children(
    trait_path: &Path,
    trait_inheritance: &HashMap<Path, HashSet<Path>>,
) -> HashSet<Path> {
    let mut res = HashSet::new();
    if let Some(supers) = trait_inheritance.get(&trait_path) {
        res.extend(supers.iter().cloned());
        for super_trait in supers.iter() {
            res.extend(union_children(super_trait, trait_inheritance).into_iter());
        }
    }
    res
}

#[proc_macro_attribute]
pub fn dync_trait(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let config: Config = syn::parse(attr).expect("Failed to parse attributes");

    let item_trait: ItemTrait =
        syn::parse(item).expect("the dync_trait attribute applies only to trait definitions");

    validate_item_trait(&item_trait);

    let mut impls = builtin_impls(&config.dync_crate_name);

    let mut trait_inheritance = builtin_trait_inheritance();

    fill_inheritance_from_item_trait(&item_trait, &mut trait_inheritance);

    fill_inherited_impls(&mut impls, &config, &trait_inheritance);

    let dync_items = construct_dync_items(&item_trait, &config, &impls, &trait_inheritance);

    let mut tokens = quote! { #item_trait };
    for item in dync_items.into_iter() {
        tokens.append_all(quote! { #item });
    }
    tokens.into()
}

// Reject unsupported item traits.
fn validate_item_trait(item_trait: &ItemTrait) {
    assert!(
        item_trait.generics.params.is_empty(),
        "trait generics are not supported by dync_trait"
    );
    assert!(
        item_trait.generics.where_clause.is_none(),
        "traits with where clauses are not supported by dync_trait"
    );
}

fn construct_dync_items(
    item_trait: &ItemTrait,
    config: &Config,
    impls: &ImplMap,
    trait_inheritance: &HashMap<Path, HashSet<Path>>,
) -> Vec<Item> {
    let crate_name = Ident::new(&config.dync_crate_name, Span::call_site());

    let trait_name = &item_trait.ident;
    let trait_path: Path = parse_quote! { #trait_name };
    let (vtable_path, _) = impls.get(&trait_path).unwrap();
    let vtable_name = &vtable_path.segments.last().unwrap().ident;
    let vis = item_trait.vis.clone();
    let super_traits = trait_inheritance.get(&trait_path).unwrap();

    // The vtable is flattened.
    let vtable_fields: Punctuated<Field, Token![,]> = super_traits
        .iter()
        .flat_map(|trait_path| {
            let (_, fns) = impls.get(&trait_path).unwrap();
            fns.iter().map(|(fn_type, _)| Field {
                attrs: Vec::new(),
                vis: Visibility::Inherited,
                ident: None,
                colon_token: None,
                ty: parse_quote! { #crate_name ::traits:: #fn_type },
            })
        })
        .collect();

    let mut has_impls: Vec<Item> = Vec::new();

    let mut fn_idx_usize = 0;
    for trait_path in super_traits.iter() {
        let (_, fns) = impls.get(&trait_path).unwrap();
        let mut methods = TokenStream::new();
        for (fn_type, fn_name) in fns.iter() {
            let fn_idx = syn::Index::from(fn_idx_usize);
            methods.append_all(quote! {
                #[inline]
                fn #fn_name ( &self ) -> &#crate_name ::traits:: #fn_type { &self.#fn_idx }
            });
            fn_idx_usize += 1;
        }
        let supertrait_name = trait_path.segments.last().unwrap().ident.clone();
        let has_trait = Ident::new(&format!("Has{}", supertrait_name), Span::call_site());

        //eprintln!("{}", &methods);

        if !trait_inheritance.contains_key(&trait_path) {
            has_impls.push(parse_quote! {
                impl #crate_name ::traits:: #has_trait for #vtable_name {
                    #methods
                }
            })
        }
    }

    let vtable_constructor = super_traits
        .iter()
        .flat_map(|trait_path| {
            let crate_name = &crate_name;
            let (_, fns) = impls.get(&trait_path).unwrap();
            let trait_ident = &trait_path.segments.last().unwrap().ident;
            let bytes_trait = Ident::new(&format!("{}Bytes", trait_ident), Span::call_site());
            fns.iter().map(move |(_, fn_name)| {
                let fn_name_string = fn_name.to_string();
                let bytes_fn = Ident::new(
                    &format!("{}_bytes", &fn_name_string[..fn_name_string.len() - 3]),
                    Span::call_site(),
                );
                let tuple: Expr =
                    parse_quote! { <T as #crate_name ::traits:: #bytes_trait> :: #bytes_fn };
                //eprintln!("{}", quote! { #tuple });
                tuple
            })
        })
        .collect::<Punctuated<Expr, Token![,]>>();

    let mut res = has_impls;
    res.push(parse_quote! {
        #[derive(Copy, Clone)]
        #vis struct #vtable_name (#vtable_fields);
    });
    res.push(parse_quote! {
        impl<T: #trait_name + 'static> #crate_name::VTable<T> for #vtable_name {
            #[inline]
            fn build_vtable() -> #vtable_name {
                #vtable_name(#vtable_constructor)
            }
        }
    });

    // Conversions to base tables
    for base_trait_path in super_traits.iter() {
        let mut conversion_exprs = Punctuated::<Expr, Token![,]>::new();

        if let Some(super_traits) = trait_inheritance.get(base_trait_path) {
            for trait_path in super_traits.iter() {
                let (_, fns) = impls.get(&trait_path).unwrap();
                for (_, fn_name) in fns.iter() {
                    conversion_exprs.push(parse_quote! { *base.#fn_name() });
                }
            }
        } else {
            let (_, fns) = impls.get(&base_trait_path).unwrap();
            for (_, fn_name) in fns.iter() {
                conversion_exprs.push(parse_quote! { *base.#fn_name() });
            }
        }

        let (base_vtable_path, _) = impls.get(&base_trait_path).unwrap();
        // super trait is a custom one, we should be able to define the conversion
        res.push(parse_quote! {
            impl From<#vtable_name> for #base_vtable_path {
                fn from(base: #vtable_name) -> Self {
                    use #crate_name :: traits::*;
                    #base_vtable_path ( #conversion_exprs )
                }
            }
        });

        //let a = res.last().unwrap();
        //eprintln!("{}", quote! { #a } );
    }

    res
}

//struct UtilityFns {
//    from_bytes_fn: ItemFn,
//    from_bytes_mut_fn: ItemFn,
//    as_bytes_fn: ItemFn,
//    box_into_box_bytes_fn: ItemFn,
//    clone_fn: (TypeBareFn, ItemFn),
//    clone_from_fn: (TypeBareFn, ItemFn),
//    clone_into_raw_fn: (TypeBareFn, ItemFn),
//    eq_fn: (TypeBareFn, ItemFn),
//    hash_fn: (TypeBareFn, ItemFn),
//    fmt_fn: (TypeBareFn, ItemFn),
//}
//
//impl UtilityFns {
//    fn new() -> Self {
//        // Byte Helpers
//        let from_bytes_fn: ItemFn = parse_quote! {
//            #[inline]
//            unsafe fn from_bytes<S: 'static>(bytes: &[u8]) -> &S {
//                assert_eq!(bytes.len(), std::mem::size_of::<S>());
//                &*(bytes.as_ptr() as *const S)
//            }
//        };
//
//        let from_bytes_mut_fn: ItemFn = parse_quote! {
//            #[inline]
//            unsafe fn from_bytes_mut<S: 'static>(bytes: &mut [u8]) -> &mut S {
//                assert_eq!(bytes.len(), std::mem::size_of::<S>());
//                &mut *(bytes.as_mut_ptr() as *mut S)
//            }
//        };
//
//        let as_bytes_fn: ItemFn = parse_quote! {
//            #[inline]
//            unsafe fn as_bytes<S: 'static>(s: &S) -> &[u8] {
//                // This is safe since any memory can be represented by bytes and we are looking at
//                // sized types only.
//                unsafe { std::slice::from_raw_parts(s as *const S as *const u8, std::mem::size_of::<S>()) }
//            }
//        };
//
//        let box_into_box_bytes_fn: ItemFn = parse_quote! {
//            #[inline]
//            fn box_into_box_bytes<S: 'static>(b: Box<S>) -> Box<[u8]> {
//                let byte_ptr = Box::into_raw(b) as *mut u8;
//                // This is safe since any memory can be represented by bytes and we are looking at
//                // sized types only.
//                unsafe { Box::from_raw(std::slice::from_raw_parts_mut(byte_ptr, std::mem::size_of::<S>())) }
//            }
//        };
//
//        // Implement known trait functions.
//        let clone_fn: (TypeBareFn, ItemFn) = (
//            parse_quote! { unsafe fn (&[u8]) -> Box<[u8]> },
//            parse_quote! {
//                #[inline]
//                unsafe fn clone_fn<S: Clone + 'static>(src: &[u8]) -> Box<[u8]> {
//                    let typed_src: &S = from_bytes(src);
//                    box_into_box_bytes(Box::new(typed_src.clone()))
//                }
//            },
//        );
//        let clone_from_fn: (TypeBareFn, ItemFn) = (
//            parse_quote! { unsafe fn (&mut [u8], &[u8]) },
//            parse_quote! {
//                #[inline]
//                unsafe fn clone_from_fn<S: Clone + 'static>(dst: &mut [u8], src: &[u8]) {
//                    let typed_src: &S = from_bytes(src);
//                    let typed_dst: &mut S = from_bytes_mut(dst);
//                    typed_dst.clone_from(typed_src);
//                }
//            },
//        );
//
//        let clone_into_raw_fn: (TypeBareFn, ItemFn) = (
//            parse_quote! { unsafe fn (&[u8], &mut [u8]) },
//            parse_quote! {
//                #[inline]
//                unsafe fn clone_into_raw_fn<S: Clone + 'static>(src: &[u8], dst: &mut [u8]) {
//                    let typed_src: &S = from_bytes(src);
//                    let cloned = S::clone(typed_src);
//                    let cloned_bytes = as_bytes(&cloned);
//                    dst.copy_from_slice(cloned_bytes);
//                    let _ = std::mem::ManuallyDrop::new(cloned);
//                }
//            },
//        );
//
//        let eq_fn: (TypeBareFn, ItemFn) = (
//            parse_quote! { unsafe fn (&[u8], &[u8]) -> bool },
//            parse_quote! {
//                #[inline]
//                unsafe fn eq_fn<S: PartialEq + 'static>(a: &[u8], b: &[u8]) -> bool {
//                    let (a, b): (&S, &S) = (from_bytes(a), from_bytes(b));
//                    a.eq(b)
//                }
//            },
//        );
//        let hash_fn: (TypeBareFn, ItemFn) = (
//            parse_quote! { unsafe fn (&[u8], &mut dyn std::hash::Hasher) },
//            parse_quote! {
//                #[inline]
//                unsafe fn hash_fn<S: std::hash::Hash + 'static>(bytes: &[u8], mut state: &mut dyn std::hash::Hasher) {
//                    let typed_data: &S = from_bytes(bytes);
//                    typed_data.hash(&mut state)
//                }
//            },
//        );
//        let fmt_fn: (TypeBareFn, ItemFn) = (
//            parse_quote! { unsafe fn (&[u8], &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> },
//            parse_quote! {
//                #[inline]
//                unsafe fn fmt_fn<S: std::fmt::Debug + 'static>(bytes: &[u8], f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
//                    let typed_data: &S = from_bytes(bytes);
//                    typed_data.fmt(f)
//                }
//            },
//        );
//
//        UtilityFns {
//            from_bytes_fn,
//            from_bytes_mut_fn,
//            as_bytes_fn,
//            box_into_box_bytes_fn,
//            clone_fn,
//            clone_from_fn,
//            clone_into_raw_fn,
//            eq_fn,
//            hash_fn,
//            fmt_fn,
//        }
//    }
//}

#[proc_macro_attribute]
pub fn dync_trait_method(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let mut trait_method: TraitItemMethod = syn::parse(item).expect(
        "the dync_trait_function attribute applies only to trait function definitions only",
    );

    trait_method.sig = dync_fn_sig(trait_method.sig);

    let tokens = quote! { #trait_method };

    tokens.into()
}

/// Convert a function signature by replacing self types with bytes.
fn dync_fn_sig(sig: Signature) -> Signature {
    assert!(
        sig.constness.is_none(),
        "const functions not supported by dync_trait"
    );
    assert!(
        sig.asyncness.is_none(),
        "async functions not supported by dync_trait"
    );
    assert!(
        sig.abi.is_none(),
        "extern functions not supported by dync_trait"
    );
    assert!(
        sig.variadic.is_none(),
        "variadic functions not supported by dync_trait"
    );

    let dync_name = format!("{}_bytes", sig.ident);
    let dync_ident = Ident::new(&dync_name, sig.ident.span());

    let mut generics = GenericsMap::new();

    // Convert generics into `dyn Trait` if possible.
    for gen in sig.generics.params.iter() {
        match gen {
            GenericParam::Type(ty) => {
                assert!(
                    ty.attrs.is_empty(),
                    "type parameter attributes are not supported by dync_trait"
                );
                assert!(
                    ty.colon_token.is_some(),
                    "unbound type parameters are not supported by dync_trait"
                );
                assert!(
                    ty.eq_token.is_none() && ty.default.is_none(),
                    "default type parameters are not supported by dync_trait"
                );
                generics.insert(ty.ident.clone(), ty.bounds.clone());
            }
            GenericParam::Lifetime(_) => {
                panic!("lifetime parameters in trait functions are not supported by dync_trait");
            }
            GenericParam::Const(_) => {
                panic!("const parameters in trait functions are not supported by dync_trait");
            }
        }
    }
    if let Some(where_clause) = sig.generics.where_clause {
        for pred in where_clause.predicates.iter() {
            match pred {
                WherePredicate::Type(ty) => {
                    assert!(
                        ty.lifetimes.is_none(),
                        "lifetimes in for bindings are not supported by dync_trait"
                    );
                    if let Type::Path(ty_path) = ty.bounded_ty.clone() {
                        assert!(
                            ty_path.qself.is_none(),
                            "complex trait bounds are not supported by dync_trait"
                        );
                        assert!(
                            ty_path.path.leading_colon.is_none(),
                            "complex trait bounds are not supported by dync_trait"
                        );
                        assert!(
                            ty_path.path.segments.len() != 1,
                            "complex trait bounds are not supported by dync_trait"
                        );
                        let seg = ty_path.path.segments.first().unwrap();
                        assert!(
                            !seg.arguments.is_empty(),
                            "complex trait bounds are not supported by dync_trait"
                        );
                        generics.insert(seg.ident.clone(), ty.bounds.clone());
                    }
                }
                WherePredicate::Lifetime(_) => {
                    panic!(
                        "lifetime parameters in trait functions are not supported by dync_trait"
                    );
                }
                _ => {}
            }
        }
    }

    // Convert inputs.
    let dync_inputs: Punctuated<FnArg, Token![,]> = sig
        .inputs
        .iter()
        .map(|fn_arg| {
            FnArg::Typed(match fn_arg {
                FnArg::Receiver(Receiver {
                    attrs,
                    reference,
                    mutability,
                    ..
                }) => {
                    let ty: Type = if let Some((_, lifetime)) = reference {
                        syn::parse(quote! { & #lifetime #mutability [u8] }.into()).unwrap()
                    } else {
                        syn::parse(quote! { #mutability Box<[u8]> }.into()).unwrap()
                    };
                    PatType {
                        attrs: attrs.to_vec(),
                        pat: syn::parse(quote! { _self_ }.into()).unwrap(),
                        colon_token: Token![:](Span::call_site()),
                        ty: Box::new(ty),
                    }
                }
                FnArg::Typed(pat_ty) => PatType {
                    ty: Box::new(type_to_bytes(process_generics(
                        *pat_ty.ty.clone(),
                        &generics,
                    ))),
                    ..pat_ty.clone()
                },
            })
        })
        .collect();

    // Convert return type.
    let dync_output: Type = match sig.output {
        ReturnType::Type(_, ty) => type_to_bytes(process_generics(*ty, &generics)),
        ReturnType::Default => syn::parse(quote! { () }.into()).unwrap(),
    };

    Signature {
        unsafety: Some(Token![unsafe](Span::call_site())),
        ident: dync_ident,
        generics: Generics {
            lt_token: None,
            params: Punctuated::new(),
            gt_token: None,
            where_clause: None,
        },
        inputs: dync_inputs,
        output: ReturnType::Type(Token![->](Span::call_site()), Box::new(dync_output)),
        ..sig
    }
}

// Translate any generics occuring in types according to the accumulated generics map by converting
// generic types into trait objects.
fn process_generics(ty: Type, generics: &GenericsMap) -> Type {
    match ty {
        Type::Paren(paren) => Type::Paren(TypeParen {
            elem: Box::new(process_generics(*paren.elem, generics)),
            ..paren
        }),
        Type::Path(path) => process_generic_type_path(path, generics, true),
        Type::Ptr(ptr) => Type::Ptr(TypePtr {
            elem: Box::new(generic_ref_to_trait_object(*ptr.elem, generics)),
            ..ptr
        }),
        Type::Reference(reference) => Type::Reference(TypeReference {
            elem: Box::new(generic_ref_to_trait_object(*reference.elem, generics)),
            ..reference
        }),
        pass_through => {
            check_for_unsupported_generics(&pass_through, generics);
            pass_through
        }
    }
}

// Convert Self type into a the given type or pass through
fn process_generic_type_path(ty: TypePath, generics: &GenericsMap, owned: bool) -> Type {
    if ty.path.leading_colon.is_some() || ty.path.segments.len() != 1 {
        return Type::Path(ty);
    }

    let seg = ty.path.segments.first().unwrap();
    if !seg.arguments.is_empty() {
        return Type::Path(ty);
    }

    // Generic types wouldn't have arguments.
    if let Some(bounds) = generics.get(&seg.ident) {
        if owned {
            syn::parse(quote! { Box<dyn #bounds> }.into()).unwrap()
        } else {
            syn::parse(quote! { dyn #bounds }.into()).unwrap()
        }
    } else {
        Type::Path(ty)
    }
}

// Convert reference or pointer to self into a reference to bytes or pass through
fn generic_ref_to_trait_object(ty: Type, generics: &GenericsMap) -> Type {
    match ty {
        Type::Path(path) => process_generic_type_path(path, generics, false),
        other => other,
    }
}

// Check if there are instances of generics in unsupported places.
fn check_for_unsupported_generics(ty: &Type, generics: &GenericsMap) {
    match ty {
        Type::Array(arr) => check_for_unsupported_generics(&arr.elem, generics),
        Type::BareFn(barefn) => {
            for input in barefn.inputs.iter() {
                check_for_unsupported_generics(&input.ty, generics);
            }
            if let ReturnType::Type(_, output_ty) = &barefn.output {
                check_for_unsupported_generics(&*output_ty, generics);
            }
        }
        Type::Group(group) => check_for_unsupported_generics(&group.elem, generics),
        Type::Paren(paren) => check_for_unsupported_generics(&paren.elem, generics),
        Type::Path(path) => {
            assert!(
                path.qself.is_none(),
                "qualified paths not supported by dync_trait"
            );
            if path.path.leading_colon.is_none() && path.path.segments.len() == 1 {
                let seg = path.path.segments.first().unwrap();
                assert!(
                    seg.arguments.is_empty() && seg.ident == "Self",
                    "using Self in this context is not supported by dync_trait"
                );
            }
        }
        Type::Ptr(ptr) => check_for_unsupported_generics(&ptr.elem, generics),
        Type::Reference(reference) => check_for_unsupported_generics(&reference.elem, generics),
        Type::Slice(slice) => check_for_unsupported_generics(&slice.elem, generics),
        Type::Tuple(tuple) => {
            for elem in tuple.elems.iter() {
                check_for_unsupported_generics(elem, generics);
            }
        }
        _ => {}
    }
}

fn type_to_bytes(ty: Type) -> Type {
    // It is quite difficult to convert occurances of Self in a function signature to the
    // corresponding byte representation because of composability of types. Each type containing
    // self must know how to convert its contents to bytes, which is completely out of the scope
    // here.
    //
    // However some builtin types (like arrays, tuples and slices) and std library types can be
    // handled.  This probably one of the reasons why trait objects don't support traits with
    // functions that take in `Self` as a parameter. We will try to relax this constraint as much
    // as we can in this function.

    match ty {
        //Type::Array(arr) => Type::Array(TypeArray {
        //    elem: Box::new(type_to_bytes(*arr.elem)),
        //    ..arr
        //}),
        //Type::Group(group) => Type::Group(TypeGroup {
        //    elem: Box::new(type_to_bytes(*group.elem),
        //    ..group
        //}),
        Type::ImplTrait(impl_trait) => Type::TraitObject(TypeTraitObject {
            // Convert `impl Trait` to `dyn Trait`.
            dyn_token: Some(Token![dyn](Span::call_site())),
            bounds: impl_trait.bounds,
        }),
        Type::Paren(paren) => Type::Paren(TypeParen {
            elem: Box::new(type_to_bytes(*paren.elem)),
            ..paren
        }),
        Type::Path(path) => {
            self_type_path_into(path, syn::parse(quote! { Box<[u8]> }.into()).unwrap())
        }
        Type::Ptr(ptr) => Type::Ptr(TypePtr {
            elem: Box::new(self_to_byte_slice(*ptr.elem)),
            ..ptr
        }),
        Type::Reference(reference) => Type::Reference(TypeReference {
            elem: Box::new(self_to_byte_slice(*reference.elem)),
            ..reference
        }),
        //Type::Slice(slice) => Type::Slice(TypeSlice {
        //    elem: Box::new(type_to_bytes(*slice.elem)),
        //    ..slice
        //}),
        //Type::Tuple(tuple) => Type::Tuple(TypeTuple {
        //    elems: elems.into_iter().map(|elem| type_to_bytes(elem)),
        //    ..tuple
        //}),
        pass_through => {
            check_for_unsupported_self(&pass_through);
            pass_through
        }
    }
}

// Convert Self type into a the given type or pass through
fn self_type_path_into(path: TypePath, into_ty: Type) -> Type {
    assert!(
        path.qself.is_none(),
        "qualified paths not supported by dync_trait"
    );
    if path.path.leading_colon.is_none() && path.path.segments.len() == 1 {
        let seg = path.path.segments.first().unwrap();
        if seg.arguments.is_empty() // Self types wouldn't have arguments.
            && seg.ident == "Self"
        {
            into_ty
        } else {
            Type::Path(path)
        }
    } else {
        Type::Path(path)
    }
}

// Convert reference or pointer to self into a reference to bytes or pass through
fn self_to_byte_slice(ty: Type) -> Type {
    match ty {
        Type::Path(path) => self_type_path_into(path, syn::parse(quote! { [u8] }.into()).unwrap()),
        other => other,
    }
}

// Check if there are instances of Self in the given type, and panic if there are.
fn check_for_unsupported_self(ty: &Type) {
    match ty {
        Type::Array(arr) => check_for_unsupported_self(&arr.elem),
        Type::BareFn(barefn) => {
            for input in barefn.inputs.iter() {
                check_for_unsupported_self(&input.ty);
            }
            if let ReturnType::Type(_, output_ty) = &barefn.output {
                check_for_unsupported_self(&*output_ty);
            }
        }
        Type::Group(group) => check_for_unsupported_self(&group.elem),
        Type::Paren(paren) => check_for_unsupported_self(&paren.elem),
        Type::Path(path) => {
            assert!(
                path.qself.is_none(),
                "qualified paths not supported by dync_trait"
            );
            if path.path.leading_colon.is_none() && path.path.segments.len() == 1 {
                let seg = path.path.segments.first().unwrap();
                assert!(
                    seg.arguments.is_empty() && seg.ident == "Self",
                    "using Self in this context is not supported by dync_trait"
                );
            }
        }
        Type::Ptr(ptr) => check_for_unsupported_self(&ptr.elem),
        Type::Reference(reference) => check_for_unsupported_self(&reference.elem),
        Type::Slice(slice) => check_for_unsupported_self(&slice.elem),
        Type::Tuple(tuple) => {
            for elem in tuple.elems.iter() {
                check_for_unsupported_self(elem);
            }
        }
        _ => {}
    }
}
