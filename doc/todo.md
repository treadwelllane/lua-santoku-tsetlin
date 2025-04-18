# Now

- Optimize for stragglers, considering cache
- Get encoder working again
- Multi-layer classifiers and encoders

# Later

- Convolution/Replicated
    - See convolution and replicated branches
    - Rename to "ensemble"
    - Re-implement convolution in ensemble style, where the user can configure
      the number of "views" each sample has (like patches in convolution). Each
      view gets a separate multi-class TM that strictly sees the view it is
      associated with. Pool view-specific results by summing (consider confidence
      weighting).
    - Consider optionally allowing weight sharing (share pointer to same locks,
      state, and actions between multiple TMs)
    - Consider implementing multi-granular as having separate multi-class TMs
      for each granularity, optionally leveraging weight-sharing.

- Support training encoder via contrastive loss, triplet loss, etc.

- Consider migrating bitmap compressor here

- When iterating bits and re-computing loss, stop after loss reaches 0. In other
  words, randomly select bits to look at, flip them, recompute loss, and if loss
  improves, decrement it. When loss is zero, stop flipping bits.

- explore applying the multigranular specificity approach to to all parameters,
  but instead of applying the parameters to to a single clause (as in
  multigranular, apply the parameters by replica)

- Implement regressor

- Throw an error if length of input bitmaps isn't correct (2 x features /
  sizeof(unsigned int)) or if features is 0

# Consider

- Can we thread this by the mc output instead of by sample? Does that make
  sense? I'm thinking it would minimize locking. Problem is, only when
  n_classes > n_cores would there be a benefit.

- Explore threading at class level and clause level. Consider different
  threading approaches during training vs one-shot predict/update.

- Benchmark single thread against CAIR implementation
- Vectorize

- Better booleanizer: If threshold levels is 1, pick the middle number, if 2,
  use the 1/3 and 2/3s break points, and so on. Consider selecting levels
  dynamically instead of by fixed increments. Mini clustering algorithm?

- Flipping input bits should be done in TM library, not required by the user.
  It's too easy to forget the complement bits.

- Pass arguments as a table

- Consider pushing train & evaluate loops to C
- Re-use threads across epochs

- Should classes in Lua always be 1 indexed? Currently they're 0 indexed.
  Indexing from 1 makes sense for multi-class, but gets odd when doing 0/1
  classification and having to increment to 1/2.

- "Anchor Knowledge," For the encoder, for a new text representation, find the
  nearest observed text representation where the encoder is confidently accurate.
  Somehow use this "well understood sample" to improve the encoding of the new
  text.

# Eventually

- Absorbing (speeds up training over time as clauses are dropped)
- Coalesced (reduces memory footprint for multi-output configurations)
    - Tried this, didn't really help
- Weighted (reduces memory footprint)
- Integer-Weighted (see weighted)
- Indexed (improves learning and classification speed)
    - Triples memory usage

- Titanic dataset
