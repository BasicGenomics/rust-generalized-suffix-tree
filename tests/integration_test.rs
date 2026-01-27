use generalized_suffix_tree;

use mediumvec::{vec32, Vec32};

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_is_suffix() {
        let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
        let s1 = vec32![1,2,3,4,5,6];
        tree.add_string(s1.clone());
        for i in 0..s1.len() {
            assert!(tree.is_suffix(&s1[i..]), "{:?} should be a suffix", &s1[i..]);
        }
        assert!(!tree.is_suffix(&[1]));
        assert!(!tree.is_suffix(&[1,2]));

        let s2 = vec32![4,5,6,7,8,9,10];
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
        let s1 = vec32![1,2,3,4,5,6];
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

        let s2 = vec32![4,5,6,7,8,9,10];
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
            tree.add_string(vec32![1,2,3,4,5,6,7,8,9]);
            tree.add_string(vec32![7,8,9,10,11,12,13,14]);
            tree.pretty_print();
            assert_eq!(tree.longest_common_substring_all(), vec32![7,8,9]);
        }
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(vec32![1,2,3,4,5,6,7,8,9]);
            tree.add_string(vec32![32,31,30,4,5,6,7,8,9,10,11,12,13,14]);
            assert_eq!(tree.longest_common_substring_all(), vec32![4,5,6,7,8,9]);
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
    fn gen_random_string(len: usize, alphabet_size: usize) -> Vec32<u64> {
        let mut s = Vec32::new();
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

}
