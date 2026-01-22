//! A Generalized Suffix Tree implementation using Ukkonen's algorithm.
mod disjoint_set;

use std::collections::HashMap;
use std::rc::Rc;


type NodeID = u64;
type IndexType = u64;
type CharType = u64;

// Special nodes.
const ROOT: NodeID = 0;
const SINK: NodeID = 1;
const INVALID: NodeID = NodeID::max_value();

/// This structure represents a slice into a shared string buffer backed by `Rc<Vec<u64>>`.
#[derive(Debug, Clone)]
struct MappedSubstring {
    data: Rc<Vec<u64>>,
    start: IndexType,
    end: IndexType,
}

impl MappedSubstring {
    fn new(data: Rc<Vec<u64>>, start: IndexType, end: IndexType) -> Self {
        Self { data, start, end }
    }

    fn is_empty(&self) -> bool {
        self.start == self.end
    }

    fn len(&self) -> IndexType {
        self.end - self.start
    }

    fn as_slice(&self) -> &[u64] {
        &self.data[self.start as usize..self.end as usize]
    }
}

/// This is a node in the tree. `transitions` represents all the possible
/// transitions from this node to other nodes, indexed by the first character
/// of the string slice that transition represents. The character needs to
/// be encoded to an index between `0..MAX_CHAR_COUNT` first.
/// `suffix_link` contains the suffix link of this node (a term used in the
/// context of Ukkonen's algorithm).
/// `substr` stores the slice of the string that the transition from the parent
/// node represents. By doing so we avoid having an explicit edge data type.
#[derive(Debug)]
struct Node {
    transitions: HashMap<CharType, NodeID>,

    suffix_link: NodeID,

    /// The slice of the string this node represents.
    substr: MappedSubstring,
}

impl Node {
    fn new(data: Rc<Vec<u64>>, start: IndexType, end: IndexType) -> Self {
        Self {
            transitions: HashMap::new(),
            suffix_link: INVALID,
            substr: MappedSubstring::new(data, start, end),
        }
    }

    fn get_suffix_link(&self) -> NodeID {
        assert!(self.suffix_link != INVALID, "Invalid suffix link");
        self.suffix_link
    }
}

/// A data structure used to store the current state during the Ukkonen's algorithm.
struct ReferencePoint {
    /// The active node.
    node: NodeID,

    /// The active point index into the current string.
    index: IndexType,
}

impl ReferencePoint {
    const fn new(node: NodeID, index: IndexType) -> Self {
        Self { node, index }
    }
}

/// This is the generalized suffix tree, implemented using Ukkonen's Algorithm.
/// One important modification to the algorithm is that this is no longer an online
/// algorithm, i.e. it only accepts strings fully provided to the suffix tree, instead
/// of being able to stream processing each string. It is not a fundamental limitation and can be supported.
///
/// # Examples
///
/// ```
/// use generalized_suffix_tree::GeneralizedSuffixTree;
/// let mut tree = GeneralizedSuffixTree::new();
/// tree.add_string(vec![1,2,3,4,5,6,7,8,9]);
/// tree.add_string(vec![7,8,9,10,11,12,13,14]);
/// println!("{:?}", tree.is_suffix(&[7,8,9]));
/// ```
#[derive(Debug)]
pub struct GeneralizedSuffixTree {
    node_storage: Vec<Node>,
    inserted_strings_count: u64,
    term: u64,
}

impl Default for GeneralizedSuffixTree {
    fn default() -> Self {
        // Set the slice of root to be [0, 1) to allow it consume one character whenever we are transitioning from sink to root.
        // No other node will ever transition to root so this won't affect anything else.
        let root_data = Rc::new(vec![0]);
        let empty = Rc::new(Vec::new());
        let mut root = Node::new(root_data.clone(), 0, 1);
        let mut sink = Node::new(empty.clone(), 0, 0);

        root.suffix_link = SINK;
        sink.suffix_link = ROOT;

        let term = u64::MAX;
        let node_storage: Vec<Node> = vec![root, sink];
        Self {
            node_storage,
            inserted_strings_count: 0,
            term,
        }
    }
}

impl GeneralizedSuffixTree {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn decrement_term(&mut self) {
        self.term -= 1;
    }

    /// Add a new string to the generalized suffix tree.
    pub fn add_string(&mut self, mut s: Vec<u64>) {
        let term = self.term;
        self.decrement_term();
        self.validate_string(&s, term);

        // Add a unique terminator character to the end of the string.
        s.push(term);

        // Wrap the string in Rc so node substrings can point into it without copying.
        let rc = Rc::new(s);
        self.inserted_strings_count += 1;
        self.process_suffixes(rc);
    }

    fn validate_string(&self, s: &[u64], term: u64) {
        assert!(s.len() <= IndexType::max_value() as usize);
        assert!(!s.iter().any(|ch| *ch > self.term), "String contains character beyond allowed range");
        assert!(!s.contains(&term), "String should not contain terminator character");
    }

    /// Find the longest common substring among all strings in the suffix.
    /// This function can be used when you already have a suffix tree built,
    /// and would need to know the longest commmon substring.
    /// It can be trivially extended to support longest common substring among
    /// `K` strings.
    #[must_use]
    pub fn longest_common_substring_all(&self) -> Vec<u64> {
        let mut disjoint_set = disjoint_set::DisjointSet::new(self.node_storage.len());

        // prev_node stores the most recent occurance of a leaf that belongs to each string.
        // We use the terminator character (which uniquely represents a string) as the key.
        let mut prev_node: HashMap<CharType, NodeID> = HashMap::new();

        // lca_cnt[v] means the total number of times that the lca of two nodes is node v.
        let mut lca_cnt: Vec<usize> = vec![0; self.node_storage.len()];

        let mut longest_str: (Vec<&MappedSubstring>, IndexType) = (Vec::new(), 0);
        let mut cur_str: (Vec<&MappedSubstring>, IndexType) = (Vec::new(), 0);
        self.longest_common_substring_all_rec(
            &mut disjoint_set,
            &mut prev_node,
            &mut lca_cnt,
            ROOT,
            &mut longest_str,
            &mut cur_str,
        );

        let mut result: Vec<u64> = Vec::new();
        for s in longest_str.0 {
            result.extend(s.as_slice().iter().map(|n| *n).collect::<Vec<u64>>());
        }
        result
    }

    /// A recursive DFS that does a couple of things in one run:
    /// - Obtain the each pair of leaves that belong to the same string and are
    ///   consecutive in DFS visits. (stored in `prev_node`)
    /// - Tarjan's Algorithm to compute the least common ancestor for each
    ///   of the above pairs. (information stored in `disjoint_set`)
    /// - Maintain the number of times an LCA lands on each node. (`lca_cnt`)
    /// This function returns a tuple:
    /// - Total number of leaves in the subtree.
    /// - Sum of all LCA counts from each node in the subtree,
    /// including the node itself.
    /// These two numbers can be used to compute the number of unique strings
    /// occured in the subtree, which can be used to check whether we found
    /// a common substring.
    /// Details of the algorithm can be found here:
    /// <https://web.cs.ucdavis.edu/~gusfield/cs224f09/commonsubstrings.pdf>
    fn longest_common_substring_all_rec<'a>(
        &'a self,
        disjoint_set: &mut disjoint_set::DisjointSet,
        prev_node: &mut HashMap<CharType, NodeID>,
        lca_cnt: &mut Vec<usize>,
        node: NodeID,
        longest_str: &mut (Vec<&'a MappedSubstring>, IndexType),
        cur_str: &mut (Vec<&'a MappedSubstring>, IndexType),
    ) -> (usize, usize) {
        let mut total_leaf = 0;
        let mut total_correction = 0;
        for target_node in self.get_node(node).transitions.values() {
            if *target_node == INVALID {
                continue;
            }
            let slice = &self.get_node(*target_node).substr;
            if slice.end as usize == slice.data.len() {
                // target_node is a leaf node.
                total_leaf += 1;
                let last_ch = slice.data[(slice.end - 1) as usize];
                if let Some(prev) = prev_node.get(&last_ch) {
                    let lca = disjoint_set.find_set(*prev as usize);
                    lca_cnt[lca as usize] += 1;
                }
                prev_node.insert(last_ch, *target_node);
            } else {
                cur_str.0.push(slice);
                cur_str.1 += slice.len();
                let result = self.longest_common_substring_all_rec(
                    disjoint_set,
                    prev_node,
                    lca_cnt,
                    *target_node,
                    longest_str,
                    cur_str,
                );
                total_leaf += result.0;
                total_correction += result.1;

                cur_str.0.pop();
                cur_str.1 -= slice.len();
            }

            disjoint_set.union(node as usize, *target_node as usize);
        }
        total_correction += lca_cnt[node as usize];
        let unique_str_cnt = total_leaf - total_correction;
        if unique_str_cnt == self.inserted_strings_count as usize {
            // This node represnets a substring that is common among all strings.
            if cur_str.1 > longest_str.1 {
                *longest_str = cur_str.clone();
            }
        }
        (total_leaf, total_correction)
    }

    /// Find the longest common substring between string `s` and the current suffix.
    /// This function allows us compute this without adding `s` to the suffix.
    #[must_use]
    pub fn longest_common_substring_with<'a>(&self, s: &'a[u64]) -> &'a [u64] {
        let mut longest_start: IndexType = 0;
        let mut longest_len: IndexType = 0;
        let mut cur_start: IndexType = 0;
        let mut cur_len: IndexType = 0;
        let mut node: NodeID = ROOT;

        let chars = s;
        let mut index = 0;
        let mut active_length = 0;
        while index < chars.len() {
            let target_node_id = self.transition(node, chars[index - active_length as usize]);
            if target_node_id != INVALID {
                let slice = &self.get_node(target_node_id).substr;
                while index != chars.len()
                    && active_length < slice.len()
                    && slice.data[(active_length + slice.start) as usize] == chars[index]
                {
                    index += 1;
                    active_length += 1;
                }

                let final_len = cur_len + active_length;
                if final_len > longest_len {
                    longest_len = final_len;
                    longest_start = cur_start;
                }

                if index == chars.len() {
                    break;
                }

                if active_length == slice.len() {
                    // We can keep following this route.
                    node = target_node_id;
                    cur_len = final_len;
                    active_length = 0;
                    continue;
                }
            }
            // There was a mismatch.
            cur_start += 1;
            if cur_start as usize > index {
                index += 1;
                continue;
            }
            // We want to follow a different path with one less character from the start.
            let suffix_link = self.get_node(node).suffix_link;
            if suffix_link != INVALID && suffix_link != SINK {
                assert!(cur_len > 0);
                node = suffix_link;
                cur_len -= 1;
            } else {
                node = ROOT;
                active_length = active_length + cur_len - 1;
                cur_len = 0;
            }
            while active_length > 0 {
                assert!(((cur_start + cur_len) as usize) < chars.len());
                let target_node_id = self.transition(node, chars[(cur_start + cur_len) as usize]);
                assert!(target_node_id != INVALID);
                let slice = &self.get_node(target_node_id).substr;
                if active_length < slice.len() {
                    break;
                }
                active_length -= slice.len();
                cur_len += slice.len();
                node = target_node_id;
            }
        }
        &s[longest_start as usize..(longest_start + longest_len) as usize]
    }

    /// Checks whether a given string `s` is a suffix in the suffix tree.
    #[must_use]
    pub fn is_suffix(&self, s: &[u64]) -> bool {
        self.is_suffix_or_substr(s, false)
    }

    /// Checks whether a given string `s` is a substring of any of the strings
    /// in the suffix tree.
    #[must_use]
    pub fn is_substr(&self, s: &[u64]) -> bool {
        self.is_suffix_or_substr(s, true)
    }

    #[must_use]
    fn is_suffix_or_substr(&self, s: &[u64], check_substr: bool) -> bool {
        assert!(!s.iter().any(|ch| *ch > self.term), "Queried string cannot contain terminator char");
        let mut node = ROOT;
        let mut index = 0;
        let chars = s;
        while index < s.len() {
            let target_node = self.transition(node, chars[index]);
            if target_node == INVALID {
                return false;
            }
            let slice = &self.get_node(target_node).substr;
            for i in slice.start..slice.end {
                if index == s.len() {
                    let is_suffix = i as usize == slice.data.len() - 1;
                    return check_substr || is_suffix;
                }
                if chars[index] != slice.data[i as usize] {
                    return false;
                }
                index += 1;
            }
            node = target_node;
        }
        let mut is_suffix = false;
        // Check whether any terminator character is reachable from `node`.
        for (&ch, _) in self.get_node(node).transitions.iter() {
            if ch > self.term {
                is_suffix = true;
                break;
            }
        }

        check_substr || is_suffix
    }

    pub fn pretty_print(&self) {
        self.print_recursive(ROOT, 0);
    }

    fn print_recursive(&self, node: NodeID, space_count: u64) {
        for target_node in self.get_node(node).transitions.values() {
            if *target_node == INVALID {
                continue;
            }
            for _ in 0..space_count {
                print!(" ");
            }
            let slice = &self.get_node(*target_node).substr;
            println!("{:?}", slice.as_slice());
            self.print_recursive(*target_node, space_count + 4);
        }
    }
    fn process_suffixes(&mut self, s: Rc<Vec<u64>>) {
        let mut active_point = ReferencePoint::new(ROOT, 0);
        for i in 0..s.len() {
            let mut cur_str = MappedSubstring::new(s.clone(), active_point.index, (i + 1) as IndexType);
            active_point = self.update(active_point.node, &cur_str);
            cur_str.start = active_point.index;
            active_point = self.canonize(active_point.node, &cur_str);
        }
    }

    fn update(&mut self, node: NodeID, cur_str: &MappedSubstring) -> ReferencePoint {
        assert!(!cur_str.is_empty());

        let mut cur_str = cur_str.clone();

        let mut oldr = ROOT;

        let mut split_str = cur_str.clone();
        split_str.end -= 1;

        let last_ch = cur_str.data[(cur_str.end - 1) as usize];

        let mut active_point = ReferencePoint::new(node, cur_str.start);

        let mut r = node;

        let mut is_endpoint = self.test_and_split(node, &split_str, last_ch, &mut r);
        while !is_endpoint {
            let str_len = cur_str.data.len() as IndexType;
            let leaf_node = self.create_node_with_slice(cur_str.data.clone(), cur_str.end - 1, str_len);
            self.set_transition(r, last_ch, leaf_node);
            if oldr != ROOT {
                self.get_node_mut(oldr).suffix_link = r;
            }
            oldr = r;
            let suffix_link = self.get_node(active_point.node).get_suffix_link();
            active_point = self.canonize(suffix_link, &split_str);
            split_str.start = active_point.index;
            cur_str.start = active_point.index;
            is_endpoint = self.test_and_split(active_point.node, &split_str, last_ch, &mut r);
        }
        if oldr != ROOT {
            self.get_node_mut(oldr).suffix_link = active_point.node;
        }
        active_point
    }

    fn test_and_split(
        &mut self,
        node: NodeID,
        split_str: &MappedSubstring,
        ch: CharType,
        r: &mut NodeID,
    ) -> bool {
        if split_str.is_empty() {
            *r = node;
            return self.transition(node, ch) != INVALID;
        }
        let first_ch = split_str.data[split_str.start as usize];

        let target_node_id = self.transition(node, first_ch);
        let target_node_slice = self.get_node(target_node_id).substr.clone();

        let split_index = target_node_slice.start + split_str.len();
        let ref_ch = target_node_slice.data[split_index as usize];

        if ref_ch == ch {
            *r = node;
            return true;
        }
        // Split target_node into two nodes by inserting r in the middle.
        *r = self.create_node_with_slice(split_str.data.clone(), split_str.start, split_str.end);
        self.set_transition(*r, ref_ch, target_node_id);
        self.set_transition(node, first_ch, *r);
        self.get_node_mut(target_node_id).substr.start = split_index;

        false
    }

    fn canonize(&mut self, mut node: NodeID, cur_str: &MappedSubstring) -> ReferencePoint {
        let mut cur_str = cur_str.clone();
        loop {
            if cur_str.is_empty() {
                return ReferencePoint::new(node, cur_str.start);
            }

            let ch = cur_str.data[cur_str.start as usize];

            let target_node = self.transition(node, ch);
            if target_node == INVALID {
                break;
            }
            let slice = &self.get_node(target_node).substr;
            if slice.len() > cur_str.len() {
                break;
            }
            cur_str.start += slice.len();
            node = target_node;
        }
        ReferencePoint::new(node, cur_str.start)
    }

    fn create_node_with_slice(&mut self, data: Rc<Vec<u64>>, start: IndexType, end: IndexType) -> NodeID {
        let node = Node::new(data, start, end);
        self.node_storage.push(node);

        (self.node_storage.len() - 1) as NodeID
    }

    fn get_node(&self, node_id: NodeID) -> &Node {
        &self.node_storage[node_id as usize]
    }

    fn get_node_mut(&mut self, node_id: NodeID) -> &mut Node {
        &mut self.node_storage[node_id as usize]
    }

    fn get_string_slice_short<'a>(&'a self, slice: &'a MappedSubstring) -> &'a [u64] {
        slice.as_slice()
    }

    fn get_string(&self, slice: &MappedSubstring) -> Vec<u64> {
        slice.as_slice().iter().map(|n| *n).collect::<Vec<u64>>()
    }

    fn collect_full_strings(&self) -> Vec<Vec<u64>> {
        let mut results: Vec<Vec<u64>> = Vec::new();
        let mut cur: Vec<u64> = Vec::new();

        fn recurse(
            this: &GeneralizedSuffixTree,
            node: NodeID,
            cur: &mut Vec<u64>,
            results: &mut Vec<Vec<u64>>,
        ) {
            for (&_ch, &target) in this.get_node(node).transitions.iter() {
                if target == INVALID {
                    continue;
                }
                let slice = &this.get_node(target).substr;
                let prev_len = cur.len();
                for &v in slice.as_slice() {
                    cur.push(v);
                }
                // If this edge reaches the end of its backing string, it's a leaf for some suffix.
                // The full original string corresponds to the suffix that starts at position 0,
                // which will produce a `cur` whose length equals the backing string length.
                if slice.end as usize == slice.data.len() {
                    if cur.len() == slice.data.len() {
                        if cur.len() >= 1 {
                            let mut s = Vec::new();
                            let end = (cur.len() - 1) as usize; // drop terminator
                            for idx in 0..end {
                                s.push(cur[idx]);
                            }
                            results.push(s);
                        }
                    }
                } else {
                    recurse(this, target, cur, results);
                }
                while cur.len() > prev_len {
                    cur.pop();
                }
            }
        }

        recurse(self, ROOT, &mut cur, &mut results);

        results
    }

    /// Collect full inserted strings along with their unique terminator
    /// characters. Returns a Vec of (terminator, string-without-terminator).
    pub fn collect_full_strings_with_terms(&self) -> Vec<(u64, Vec<u64>)> {
        let mut results: Vec<(u64, Vec<u64>)> = Vec::new();
        let mut cur: Vec<u64> = Vec::new();

        fn recurse(
            this: &GeneralizedSuffixTree,
            node: NodeID,
            cur: &mut Vec<u64>,
            results: &mut Vec<(u64, Vec<u64>)>,
        ) {
            for (&_ch, &target) in this.get_node(node).transitions.iter() {
                if target == INVALID {
                    continue;
                }
                let slice = &this.get_node(target).substr;
                let prev_len = cur.len();
                for &v in slice.as_slice() {
                    cur.push(v);
                }
                if slice.end as usize == slice.data.len() {
                    if cur.len() >= 1 {
                        let term = cur[cur.len() - 1];
                        let mut s = Vec::new();
                        let end = cur.len() - 1;
                        for idx in 0..end {
                            s.push(cur[idx]);
                        }
                        results.push((term, s));
                    }
                } else {
                    recurse(this, target, cur, results);
                }
                while cur.len() > prev_len {
                    cur.pop();
                }
            }
        }

        recurse(self, ROOT, &mut cur, &mut results);

        // Keep only the longest string per terminator (the full inserted string).
        let mut map: HashMap<u64, Vec<u64>> = HashMap::new();
        for (term, s) in results.into_iter() {
            let entry = map.entry(term).or_insert_with(Vec::new);
            if s.len() > entry.len() {
                *entry = s;
            }
        }
        map.into_iter().collect()
    }

    /// Return all ordered pairs of distinct inserted full-strings that have a
    /// non-empty suffix/prefix overlap. Each returned tuple is
    /// (string_i, string_j, overlap) where `overlap` is the maximal suffix of
    /// `string_i` that equals a prefix of `string_j` (length >= 1).
    pub fn overlapping_pairs(&self) -> impl Iterator<Item = (Vec<u64>, Vec<u64>, Vec<u64>)> {
        struct OverlapIter {
            strs: Vec<Vec<u64>>,
            i: usize,
            j: usize,
        }

        impl Iterator for OverlapIter {
            type Item = (Vec<u64>, Vec<u64>, Vec<u64>);

            fn next(&mut self) -> Option<Self::Item> {
                let n = self.strs.len();
                while self.i < n {
                    if self.j >= n {
                        self.i += 1;
                        self.j = 0;
                        continue;
                    }
                    if self.i == self.j {
                        self.j += 1;
                        continue;
                    }
                    let si = &self.strs[self.i];
                    let sj = &self.strs[self.j];
                    let len_i = si.len() as usize;
                    let len_j = sj.len() as usize;
                    let mut best_k = 0usize;
                    let max_k = std::cmp::min(len_i, len_j);
                    for k in 1..=max_k {
                        let mut ok = true;
                        for t in 0..k {
                            if si[len_i - k + t] != sj[t] {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            best_k = k; // keep largest found
                        }
                    }
                    self.j += 1;
                    if best_k > 0 {
                        let mut overlap = Vec::new();
                        let start = len_i - best_k;
                        for t in start..len_i {
                            overlap.push(si[t]);
                        }
                        return Some((si.clone(), sj.clone(), overlap));
                    }
                }
                None
            }
        }

        OverlapIter { strs: self.collect_full_strings(), i: 0, j: 0 }
    }

    /// Streaming iterator that yields tuples `(term_i, term_j, overlap_vec)`
    /// where `term_i` and `term_j` are the unique terminator characters that
    /// identify the original inserted strings, and `overlap_vec` is the
    /// overlapping sequence (suffix of i == prefix of j) with length >= 1.
    pub fn overlapping_pairs_indices(&self) -> impl Iterator<Item = (u64, u64, Vec<u64>)> {
        struct OverlapIdxIter {
            pairs: Vec<(u64, Vec<u64>)>,
            i: usize,
            j: usize,
        }

        impl Iterator for OverlapIdxIter {
            type Item = (u64, u64, Vec<u64>);

            fn next(&mut self) -> Option<Self::Item> {
                let n = self.pairs.len();
                while self.i < n {
                    if self.j >= n {
                        self.i += 1;
                        self.j = 0;
                        continue;
                    }
                    if self.i == self.j {
                        self.j += 1;
                        continue;
                    }
                    let (term_i, ref si) = &self.pairs[self.i];
                    let (term_j, ref sj) = &self.pairs[self.j];
                    let len_i = si.len();
                    let len_j = sj.len();
                    let mut best_k = 0usize;
                    let max_k = std::cmp::min(len_i, len_j);
                    for k in 1..=max_k {
                        let mut ok = true;
                        for t in 0..k {
                            if si[len_i - k + t] != sj[t] {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            best_k = k;
                        }
                    }
                    self.j += 1;
                    if best_k > 0 {
                        let mut overlap = Vec::new();
                        let start = len_i - best_k;
                        for t in start..len_i {
                            overlap.push(si[t]);
                        }
                        return Some((*term_i, *term_j, overlap));
                    }
                }
                None
            }
        }

        OverlapIdxIter { pairs: self.collect_full_strings_with_terms(), i: 0, j: 0 }
    }

    /// Streaming iterator that yields `(term_i, term_j, overlap_slice)` where
    /// `overlap_slice` is a borrowed slice into the iterator's internal buffer
    /// (no allocation per yielded overlap).
    /// Streaming iterator that yields `(term_i, term_j, start, len)` where
    /// `start` and `len` are offsets into the full string corresponding to
    /// `term_i` describing the overlap region. This avoids allocating overlap
    /// vectors per yielded pair; consumers can map terminators to strings via
    /// `collect_full_strings_with_terms()` and slice accordingly.
    pub fn overlapping_pairs_indices_noalloc(&self) -> impl Iterator<Item = (u64, u64, usize, usize)> {
        struct OverlapIdxNoAllocIter {
            pairs: Vec<(u64, Vec<u64>)>,
            i: usize,
            j: usize,
        }

        impl Iterator for OverlapIdxNoAllocIter {
            type Item = (u64, u64, usize, usize);

            fn next(&mut self) -> Option<Self::Item> {
                let n = self.pairs.len();
                while self.i < n {
                    if self.j >= n {
                        self.i += 1;
                        self.j = 0;
                        continue;
                    }
                    if self.i == self.j {
                        self.j += 1;
                        continue;
                    }
                    let (term_i, ref si) = &self.pairs[self.i];
                    let (term_j, ref sj) = &self.pairs[self.j];
                    let len_i = si.len();
                    let len_j = sj.len();
                    let mut best_k = 0usize;
                    let max_k = std::cmp::min(len_i, len_j);
                    for k in 1..=max_k {
                        let mut ok = true;
                        for t in 0..k {
                            if si[len_i - k + t] != sj[t] {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            best_k = k;
                        }
                    }
                    self.j += 1;
                    if best_k > 0 {
                        let start = len_i - best_k;
                        return Some((*term_i, *term_j, start, best_k));
                    }
                }
                None
            }
        }

        OverlapIdxNoAllocIter { pairs: self.collect_full_strings_with_terms(), i: 0, j: 0 }
    }

    /// Collect leaf nodes producing the unique terminator -> backing Rc mapping.
    fn collect_leaf_nodes_with_terms(&self) -> Vec<(u64, Rc<Vec<u64>>)> {
        let mut map: HashMap<u64, Rc<Vec<u64>>> = HashMap::new();

        fn recurse(
            this: &GeneralizedSuffixTree,
            node: NodeID,
            map: &mut HashMap<u64, Rc<Vec<u64>>>,
        ) {
            for (&_ch, &target) in this.get_node(node).transitions.iter() {
                if target == INVALID {
                    continue;
                }
                let slice = &this.get_node(target).substr;
                if slice.end as usize == slice.data.len() {
                    // leaf
                    if slice.end > 0 {
                        let term = slice.data[(slice.end - 1) as usize];
                        let rc = slice.data.clone();
                        let entry = map.entry(term).or_insert_with(|| rc.clone());
                        if rc.len() > entry.len() {
                            *entry = rc.clone();
                        }
                    }
                } else {
                    recurse(this, target, map);
                }
            }
        }

        recurse(self, ROOT, &mut map);
        map.into_iter().collect()
    }

    /// Streaming iterator that yields `(term_i, term_j, rc_for_i, start, len)`
    /// where `rc_for_i` is an `Rc<Vec<u64>>` referencing the original
    /// backing buffer for the i'th string (no cloning of string contents).
    pub fn overlapping_pairs_nodes(&self) -> impl Iterator<Item = (u64, u64, Rc<Vec<u64>>, usize, usize)> {
        struct OverlapNodeIter {
            pairs: Vec<(u64, Rc<Vec<u64>>)>,
            i: usize,
            j: usize,
        }

        impl Iterator for OverlapNodeIter {
            type Item = (u64, u64, Rc<Vec<u64>>, usize, usize);

            fn next(&mut self) -> Option<Self::Item> {
                let n = self.pairs.len();
                while self.i < n {
                    if self.j >= n {
                        self.i += 1;
                        self.j = 0;
                        continue;
                    }
                    if self.i == self.j {
                        self.j += 1;
                        continue;
                    }
                    let (term_i, ref rc_i) = &self.pairs[self.i];
                    let (term_j, ref rc_j) = &self.pairs[self.j];
                    let len_i = rc_i.len();
                    let len_j = rc_j.len();
                    // Exclude terminator at the end of each buffer when matching.
                    if len_i == 0 || len_j == 0 {
                        self.j += 1;
                        continue;
                    }
                    let data_len_i = len_i - 1;
                    let data_len_j = len_j - 1;
                    let mut best_k = 0usize;
                    let max_k = std::cmp::min(data_len_i, data_len_j);
                    for k in 1..=max_k {
                        let mut ok = true;
                        for t in 0..k {
                            if rc_i[data_len_i - k + t] != rc_j[t] {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            best_k = k;
                        }
                    }
                    self.j += 1;
                    if best_k > 0 {
                        let start = data_len_i - best_k;
                        return Some((*term_i, *term_j, rc_i.clone(), start, best_k));
                    }
                }
                None
            }
        }

        // Build pairs from collected full strings with terms. This currently
        // wraps each collected `Vec<u64>` into an `Rc` so the iterator can
        // return an `Rc<Vec<u64>>`. This avoids walking nodes directly and
        // is simpler and robust; it can be optimized later to avoid the
        // allocation if desired.
        let pairs_rc: Vec<(u64, Rc<Vec<u64>>)> = self
            .collect_full_strings_with_terms()
            .into_iter()
            .map(|(t, mut s)| {
                s.push(t);
                (t, Rc::new(s))
            })
            .collect();

        OverlapNodeIter { pairs: pairs_rc, i: 0, j: 0 }
    }

    fn transition(&self, node: NodeID, ch: CharType) -> NodeID {
        if node == SINK {
            // SINK always transition to ROOT.
            return ROOT;
        }
        match self.get_node(node).transitions.get(&ch) {
            None => INVALID,
            Some(x) => *x,
        }
    }

    fn set_transition(&mut self, node: NodeID, ch: CharType, target_node: NodeID) {
        self.get_node_mut(node).transitions.insert(ch, target_node);
    }

    
}
