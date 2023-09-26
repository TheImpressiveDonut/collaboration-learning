### Some changes

why named ref ?
why named PDLDataset ?

### TODO

- [ ] refactor dataset generation:
  - refactor cifar 10 and 100 in one file
  - refactor gen args in one file
  - simplify and improve readability of utils dataset function
  - use numpy when possible instead of python loops or list
  - replace fed isic manual download to auto
- [ ] update env collab-consensus
- [x] refactor client
  - should separate client from loading data
- [ ] why ref name ?
- [ ] check memory usage
- [ ] check profiling
- [x] hash config for multiple dataset