use proc_macro2::{Span, TokenStream};
use quote::{quote, TokenStreamExt};
use std::collections::HashMap;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::*;

type GenericsMap = HashMap<Ident, Punctuated<TypeParamBound, Token![+]>>;

#[derive(Debug)]
struct Config {
    dyn_crate_name: String,
    suffix: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            dyn_crate_name: String::from("dyn"),
            suffix: String::from("Bytes"),
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
        let attribs: Punctuated<DynAttrib, Token![,]> =
            Punctuated::parse_separated_nonempty(input)?;
        for attrib in attribs.iter() {
            let name = attrib.ident.to_string();
            match (name.as_str(), &attrib.value) {
                ("dyn_crate_name", Some(Lit::Str(ref lit))) => {
                    config.dyn_crate_name = lit.value().clone()
                }
                ("suffix", Some(Lit::Str(ref lit))) => config.suffix = lit.value().clone(),
                _ => {}
            }
        }
        Ok(config)
    }
}

#[proc_macro_attribute]
pub fn dyn_trait(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let config: Config = syn::parse(attr).expect("Failed to parse attributes");

    let item_trait: ItemTrait =
        syn::parse(item).expect("the dyn_trait attribute applies only to trait definitions");

    let dyn_items = construct_dyn_items(&item_trait, &config);

    let tokens = quote! {
        #item_trait

        #dyn_items
    };

    tokens.into()
}

fn construct_dyn_items(item_trait: &ItemTrait, config: &Config) -> TokenStream {
    assert!(
        item_trait.generics.params.is_empty(),
        "trait generics are not supported by dyn_trait"
    );
    assert!(
        item_trait.generics.where_clause.is_none(),
        "traits with where clauses are not supported by dyn_trait"
    );

    // Implement known trait functions.
    let clone_fn: (TypeBareFn, ItemFn) = (
        parse_quote! { unsafe fn (&[u8]) -> Box<[u8]> },
        parse_quote! {
            unsafe fn clone_fn<S: Clone + 'static>(src: &[u8]) -> Box<[u8]> {
                let typed_src: &S = Bytes::from_bytes(src);
                Bytes::box_into_box_bytes(Box::new(typed_src.clone()))
            }
        }
    );
    let clone_from_fn: (TypeBareFn, ItemFn) = (
        parse_quote! { unsafe fn (&mut [u8], &[u8]) },
        parse_quote! {
            unsafe fn clone_from_fn<S: Clone + 'static>(dst: &mut [u8], src: &[u8]) {
                let typed_src: &S = Bytes::from_bytes(src);
                let typed_dst: &mut S = Bytes::from_bytes_mut(dst);
                typed_dst.clone_from(typed_src);
            }
        }
    );
    let eq_fn: (TypeBareFn, ItemFn) = (
        parse_quote! { unsafe fn (&[u8], &[u8]) -> bool },
        parse_quote! {
            unsafe fn eq_fn<S: PartialEq + 'static>(a: &[u8], b: &[u8]) -> bool {
                let (a, b): (&S, &S) = (Bytes::from_bytes(a), Bytes::from_bytes(b));
                a.eq(b)
            }
        }
    );
    let hash_fn: (TypeBareFn, ItemFn) = (
        parse_quote! { unsafe fn (&[u8], &mut dyn std::hash::Hasher) },
        parse_quote! {
            unsafe fn hash_fn<S: std::hash::Hash + 'static>(bytes: &[u8], mut state: &mut dyn std::hash::Hasher) {
                let typed_data: &S = Bytes::from_bytes(bytes);
                typed_data.hash(&mut state)
            }
        }
    );
    let fmt_fn: (TypeBareFn, ItemFn) = (
        parse_quote! { unsafe fn (&[u8], &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> },
        parse_quote! {
            unsafe fn fmt_fn<S: std::fmt::Debug + 'static>(bytes: &[u8], f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
                let typed_data: &S = Bytes::from_bytes(bytes);
                typed_data.fmt(f)
            }
        }
    );

    let mut known_traits: HashMap<Path, Vec<(TypeBareFn, ItemFn)>> = HashMap::new();
    known_traits.insert(parse_quote! { Clone }, vec![clone_fn, clone_from_fn]);
    known_traits.insert(parse_quote! { PartialEq }, vec![eq_fn]);
    known_traits.insert(parse_quote! { std::hash::Hash }, vec![hash_fn]);
    known_traits.insert(parse_quote! { std::fmt::Debug }, vec![fmt_fn]);

    let trait_name = item_trait.ident.clone();
    let vtable_name = Ident::new(&format!("{}{}", &trait_name, config.suffix), Span::call_site());

    let vis = item_trait.vis.clone();

    let vtable: Vec<_> = item_trait.supertraits.iter().filter_map(|bound| {
        match bound {
            TypeParamBound::Trait(bound) => {
                if bound.lifetimes.is_some()
                    || bound.modifier != TraitBoundModifier::None
                {
                    // We are looking for recognizable traits only
                    None
                } else {
                    let seg = bound.path.segments.first().unwrap();
                    if !seg.arguments.is_empty() {
                        None
                    } else {
                        known_traits.get_key_value(&bound.path).map(|(path, table)| 
                            (path.clone(), table.clone())
                        )
                    }
                }
            }
            _ => None,
        }
    }).collect();

    let vtable_fields: Punctuated<Field, Token![,]> = vtable.iter().map(|(_, table)| {
        let fns: Punctuated<Type, Token![,]> = table.iter().map(|(ty, _)| Type::BareFn(ty.clone())).collect();
        Field {
            attrs: Vec::new(),
            vis: Visibility::Inherited,
            ident: None,
            colon_token: None,
            ty: parse_quote! { (#fns) },
        }
    }).collect();

    let mut has_impls = TokenStream::new();
    for (table_idx_usize, (path, table)) in vtable.iter().enumerate() {
        let table_idx = syn::Index::from(table_idx_usize);
        let mut methods = TokenStream::new();
        if table.len() == 1 {
            let (fn_type, fn_def) = table.first().unwrap();
            let fn_name = &fn_def.sig.ident;
            methods.append_all(quote!{
                #[inline]
                fn #fn_name ( &self ) -> &#fn_type { &self.#table_idx }
            });
        } else {
            for (fn_idx_usize, (fn_type, fn_def)) in table.iter().enumerate() {
                let fn_idx = syn::Index::from(fn_idx_usize);
                let fn_name = &fn_def.sig.ident;
                methods.append_all(quote!{
                    #[inline]
                    fn #fn_name ( &self ) -> &#fn_type { &(self.#table_idx).#fn_idx }
                });
            }
        }

        let supertrait_name = path.segments.last().unwrap().ident.clone();
        let has_trait = Ident::new(&format!("Has{}", supertrait_name), Span::call_site());

        has_impls.append_all(quote! {
            impl #has_trait for #vtable_name {
                #methods
            }
        })
    }

    let vtable_constructor = vtable.iter().map(|(_, fntable)| {
        let fields = fntable.iter().map(|(_, fn_def)| {
            let fn_name = fn_def.sig.ident.clone();
            let expr: Expr = parse_quote! { #fn_name::<T> };
            expr
        }).collect::<Punctuated<Expr, Token![,]>>();
        let tuple: Expr = if fields.len() > 1 {
            parse_quote! { (#fields) }
        } else {
            parse_quote! { #fields }
        };
        tuple
    }).collect::<Punctuated<Expr, Token![,]>>();

    let crate_name = Ident::new(&config.dyn_crate_name, Span::call_site());

    let fns_defs = vtable.iter().flat_map(|(_, fntable)| {
        fntable.iter().map(|(_, fn_def)| {
            parse_quote! { #fn_def }
        })
    }).collect::<Vec<Stmt>>();

    let mut build_vtable_block = TokenStream::new();
    for fn_def in fns_defs.iter() {
        build_vtable_block.append_all(quote! { #fn_def });
    }

    build_vtable_block.append_all(quote! { #vtable_name(#vtable_constructor) });

    quote! {
        #vis struct #vtable_name (#vtable_fields);
        
        #has_impls

        impl<T: #trait_name + 'static> #crate_name::Dyn for T {
            type VTable = #vtable_name;
            #[inline]
            fn build_vtable() -> Self::VTable {
                #build_vtable_block
            }
        }
    }
}

#[proc_macro_attribute]
pub fn dyn_trait_method(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let mut trait_method: TraitItemMethod = syn::parse(item)
        .expect("the dyn_trait_function attribute applies only to trait function definitions only");

    trait_method.sig = dyn_fn_sig(trait_method.sig);

    let tokens = quote! { #trait_method };

    tokens.into()
}

/// Convert a function signature by replacing self types with bytes.
fn dyn_fn_sig(sig: Signature) -> Signature {
    assert!(
        sig.constness.is_none(),
        "const functions not supported by dyn_trait"
    );
    assert!(
        sig.asyncness.is_none(),
        "async functions not supported by dyn_trait"
    );
    assert!(
        sig.abi.is_none(),
        "extern functions not supported by dyn_trait"
    );
    assert!(
        sig.variadic.is_none(),
        "variadic functions not supported by dyn_trait"
    );

    let dyn_name = format!("{}_bytes", sig.ident);
    let dyn_ident = Ident::new(&dyn_name, sig.ident.span().clone());

    let mut generics = GenericsMap::new();

    // Convert generics into `dyn Trait` if possible.
    for gen in sig.generics.params.iter() {
        match gen {
            GenericParam::Type(ty) => {
                assert!(
                    ty.attrs.is_empty(),
                    "type parameter attributes are not supported by dyn_trait"
                );
                assert!(
                    ty.colon_token.is_some(),
                    "unbound type parameters are not supported by dyn_trait"
                );
                assert!(
                    ty.eq_token.is_none() && ty.default.is_none(),
                    "default type parameters are not supported by dyn_trait"
                );
                generics.insert(ty.ident.clone(), ty.bounds.clone());
            }
            GenericParam::Lifetime(_) => {
                panic!("lifetime parameters in trait functions are not supported by dyn_trait");
            }
            GenericParam::Const(_) => {
                panic!("const parameters in trait functions are not supported by dyn_trait");
            }
        }
    }
    if let Some(where_clause) = sig.generics.where_clause {
        for pred in where_clause.predicates.iter() {
            match pred {
                WherePredicate::Type(ty) => {
                    assert!(
                        ty.lifetimes.is_none(),
                        "lifetimes in for bindings are not supported by dyn_trait"
                    );
                    if let Type::Path(ty_path) = ty.bounded_ty.clone() {
                        assert!(
                            ty_path.qself.is_none(),
                            "complex trait bounds are not supported by dyn_trait"
                        );
                        assert!(
                            ty_path.path.leading_colon.is_none(),
                            "complex trait bounds are not supported by dyn_trait"
                        );
                        assert!(
                            ty_path.path.segments.len() != 1,
                            "complex trait bounds are not supported by dyn_trait"
                        );
                        let seg = ty_path.path.segments.first().unwrap();
                        assert!(
                            !seg.arguments.is_empty(),
                            "complex trait bounds are not supported by dyn_trait"
                        );
                        generics.insert(seg.ident.clone(), ty.bounds.clone());
                    }
                }
                WherePredicate::Lifetime(_) => {
                    panic!("lifetime parameters in trait functions are not supported by dyn_trait");
                }
                _ => {}
            }
        }
    }

    // Convert inputs.
    let dyn_inputs: Punctuated<FnArg, Token![,]> = sig
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
    let dyn_output: Type = match sig.output {
        ReturnType::Type(_, ty) => type_to_bytes(process_generics(*ty, &generics)),
        ReturnType::Default => syn::parse(quote! { () }.into()).unwrap(),
    };

    Signature {
        unsafety: Some(Token![unsafe](Span::call_site())),
        ident: dyn_ident,
        generics: Generics {
            lt_token: None,
            params: Punctuated::new(),
            gt_token: None,
            where_clause: None,
        },
        inputs: dyn_inputs,
        output: ReturnType::Type(Token![->](Span::call_site()), Box::new(dyn_output)),
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
                "qualified paths not supported by dyn_trait"
            );
            if path.path.leading_colon.is_none() && path.path.segments.len() == 1 {
                let seg = path.path.segments.first().unwrap();
                assert!(
                    seg.arguments.is_empty() && "Self".to_string() == seg.ident.to_string(),
                    "using Self in this context is not supported by dyn_trait"
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
        "qualified paths not supported by dyn_trait"
    );
    if path.path.leading_colon.is_none() && path.path.segments.len() == 1 {
        let seg = path.path.segments.first().unwrap();
        if seg.arguments.is_empty() // Self types wouldn't have arguments.
            && "Self".to_string() == seg.ident.to_string()
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
    let res = match ty {
        Type::Path(path) => self_type_path_into(path, syn::parse(quote! { [u8] }.into()).unwrap()),
        other => other,
    };
    res
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
                "qualified paths not supported by dyn_trait"
            );
            if path.path.leading_colon.is_none() && path.path.segments.len() == 1 {
                let seg = path.path.segments.first().unwrap();
                assert!(
                    seg.arguments.is_empty() && "Self".to_string() == seg.ident.to_string(),
                    "using Self in this context is not supported by dyn_trait"
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
