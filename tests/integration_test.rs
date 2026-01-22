use generalized_suffix_tree;

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_is_suffix() {
        let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
        let s1 = vec![1,2,3,4,5,6];
        tree.add_string(s1.clone());
        for i in 0..s1.len() {
            assert!(tree.is_suffix(&s1[i..]), "{:?} should be a suffix", &s1[i..]);
        }
        assert!(!tree.is_suffix(&[1]));
        assert!(!tree.is_suffix(&[1,2]));

        let s2 = vec![4,5,6,7,8,9,10];
        tree.add_string(s2.clone());
        for i in 0..s1.len() {
            assert!(tree.is_suffix(&s1[i..]), "{:?} should be a suffix", &s1[i..]);
        }
        for i in 0..s2.len() {
            assert!(tree.is_suffix(&s2[i..]), "{:?} should be a suffix", &s2[i..]);
        }
        assert!(!tree.is_suffix(&[2,3]));
    }

    #[test]
    fn test_is_substr() {
        let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
        let s1 = vec![1,2,3,4,5,6];
        tree.add_string(s1.clone());
        for i in 0..s1.len() {
            for j in i..s1.len() {
                assert!(
                    tree.is_substr(&s1[i..(j + 1)]),
                    "{:?} should be a substring",
                    &s1[i..(j + 1)]
                );
            }
        }
        assert!(!tree.is_substr(&[2,3,10]));
        assert!(!tree.is_substr(&[3,4,50]));

        let s2 = vec![4,5,6,7,8,9,10];
        tree.add_string(s2.clone());
        for i in 0..s1.len() {
            for j in i..s1.len() {
                assert!(
                    tree.is_substr(&s1[i..(j + 1)]),
                    "{:?} should be a substring",
                    &s1[i..(j + 1)]
                );
            }
        }
        for i in 0..s2.len() {
            for j in i..s2.len() {
                assert!(
                    tree.is_substr(&s2[i..(j + 1)]),
                    "{:?} should be a substring",
                    &s2[i..(j + 1)]
                );
            }
        }
        assert!(!tree.is_suffix(&[2,3]));
    }

    #[test]
    fn test_longest_common_substring_all() {
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(vec![1,2,3,4,5,6,7,8,9]);
            tree.add_string(vec![7,8,9,10,11,12,13,14]);
            tree.pretty_print();
            assert_eq!(tree.longest_common_substring_all(), vec![7,8,9]);
        }
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(vec![1,2,3,4,5,6,7,8,9]);
            tree.add_string(vec![32,31,30,4,5,6,7,8,9,10,11,12,13,14]);
            assert_eq!(tree.longest_common_substring_all(), vec![4,5,6,7,8,9]);
        }
    }
/*
    #[test]
    fn test_longest_common_substring_with() {
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(String::from("VOTEFORTHEGREATALBANIAFORYOU"), '$');
            let test_str = String::from("CHOOSETHEGREATALBANIANFUTURE");
            assert_eq!(
                tree.longest_common_substring_with(&test_str),
                "THEGREATALBANIA"
            );
            tree.add_string(test_str, '#');
            let test_str = String::from("VOTECHOOSEGREATALBANIATHEFUTURE");
            assert_eq!(
                tree.longest_common_substring_with(&test_str),
                "EGREATALBANIA"
            );
        }
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(String::from("HHDBBCIAAE"), '$');
            let test_str = String::from("AAFJEHDAEG");
            assert_eq!(tree.longest_common_substring_with(&test_str).len(), 2);
        }
    }
 */
    fn gen_random_string(len: usize, alphabet_size: usize) -> Vec<u64> {
        let mut s = Vec::new();
        for _ in 0..len {
            let ch = (rand::random::<u64>() % alphabet_size as u64);
            s.push(ch as u64);
        }
        s
    }

    #[test]
    #[ignore]
    fn test_longest_common_substring_cross_check() {
        for _ in 0..10000 {
            let s1 = gen_random_string(100, 10);
            let s2 = gen_random_string(100, 10);
            let result1 = {
                let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
                tree.add_string(s1.clone());
                tree.add_string(s2.clone());
                tree.longest_common_substring_all()
            };
            let result2 = {
                let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
                tree.add_string(s1.clone());
                tree.longest_common_substring_with(&s2)
            };
            let result3 = {
                let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
                tree.add_string(s2.clone());
                tree.longest_common_substring_with(&s1)
            };
            assert_eq!(result1.len(), result2.len());
            assert_eq!(result1.len(), result3.len());
        }
    }

    #[test]
    fn test_overlapping_pairs() {
        let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
        tree.add_string(vec![1,2,3]);
        tree.add_string(vec![2,3,4]);
        tree.add_string(vec![1,2,5]);
        tree.add_string(vec![2,5,6]);

        let pairs: Vec<_> = tree.overlapping_pairs().collect();

        // Expect (1,2,3) overlaps (2,3,4) on [2,3]
        assert!(pairs.iter().any(|(a,b,ov)| a == &vec![1,2,3] && b == &vec![2,3,4] && ov == &vec![2,3]));

        // (1,2,3) should NOT overlap (1,2,5)
        assert!(!pairs.iter().any(|(a,b,_)| a == &vec![1,2,3] && b == &vec![1,2,5]));

        // (1,2,5) overlaps (2,5,6) on [2,5]
        assert!(pairs.iter().any(|(a,b,ov)| a == &vec![1,2,5] && b == &vec![2,5,6] && ov == &vec![2,5]));
    }

    #[test]
    fn test_overlapping_indices() {
        let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
        tree.add_string(vec![1,2,3]);
        tree.add_string(vec![2,3,4]);
        tree.add_string(vec![1,2,5]);
        tree.add_string(vec![2,5,6]);

        let pairs: Vec<_> = tree.overlapping_pairs_indices().collect();

        // There should be overlaps [2,3] and [2,5]
        assert!(pairs.iter().any(|(_,_,ov)| ov == &vec![2,3]));
        assert!(pairs.iter().any(|(_,_,ov)| ov == &vec![2,5]));

        // Verify that (1,2,3) does not overlap (1,2,5) by mapping terms
        let terms = tree.collect_full_strings_with_terms();
        let term_123 = terms.iter().find(|(_t,s)| s == &vec![1,2,3]).unwrap().0;
        let term_125 = terms.iter().find(|(_t,s)| s == &vec![1,2,5]).unwrap().0;
        assert!(!pairs.iter().any(|(a,b,_)| *a == term_123 && *b == term_125));
    }

    #[test]
    fn test_overlapping_indices_noalloc() {
        let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
        tree.add_string(vec![1,2,3]);
        tree.add_string(vec![2,3,4]);
        tree.add_string(vec![1,2,5]);
        tree.add_string(vec![2,5,6]);

        let pairs: Vec<_> = tree.overlapping_pairs_indices_noalloc().collect();

        let terms = tree.collect_full_strings_with_terms();
        let mut map = std::collections::HashMap::new();
        for (t, s) in &terms {
            map.insert(*t, s.clone());
        }

        // Check presence of overlap [2,3]
        assert!(pairs.iter().any(|(a,b,start,len)| {
            if let (Some(sa), Some(_sb)) = (map.get(a), map.get(b)) {
                let mut got = Vec::new();
                for idx in *start..(*start + *len) {
                    got.push(sa[idx]);
                }
                got == vec![2,3]
            } else { false }
        }));

        // Check presence of overlap [2,5]
        assert!(pairs.iter().any(|(a,b,start,len)| {
            if let (Some(sa), Some(_sb)) = (map.get(a), map.get(b)) {
                let mut got = Vec::new();
                for idx in *start..(*start + *len) {
                    got.push(sa[idx]);
                }
                got == vec![2,5]
            } else { false }
        }));
    }

    #[test]
    fn test_overlapping_pairs_nodes() {
        let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
        tree.add_string(vec![1,2,3]);
        tree.add_string(vec![2,3,4]);
        tree.add_string(vec![1,2,5]);
        tree.add_string(vec![2,5,6]);

        let _terms = tree.collect_full_strings_with_terms();
        let pairs: Vec<_> = tree.overlapping_pairs_nodes().collect();

        // Ensure we actually found some pairs
        assert!(!pairs.is_empty(), "no pairs were found by overlapping_pairs_nodes");

        for (a, b, rc, start, len) in &pairs {
            println!("Overlap between {:?} and {:?}: {:?} (start {}, len {})", a, b, &rc[*start..(*start + *len)], start, len);
        }

        // Check that we observe overlap [2,3]
        assert!(pairs.iter().any(|(_a,_b,rc,start,len)| {
            let mut got = Vec::new();
            for idx in *start..(*start + *len) {
                got.push(rc[idx]);
            }
            got == vec![2,3]
        }));

        // Check that we observe overlap [2,5]
        assert!(pairs.iter().any(|(_a,_b,rc,start,len)| {
            let mut got = Vec::new();
            for idx in *start..(*start + *len) {
                got.push(rc[idx]);
            }
            got == vec![2,5]
        }));
    }
}
