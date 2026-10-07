# Sample files for the Upload flow

Two synthetic files to try **Use your own files** with. Every voter in them is invented: see `modules/voter_contact.py`.

| File | Rows | What it is |
|---|---|---|
| `voters_10k_A.csv` | 10,000 | Dataset A: 5% of voters appear twice (a re-registration, sometimes after a move) |
| `voters_10k_B.csv` | 5,000 | Dataset B: a damaged sample of A (typos, shifted dates, blanks, a changed phone digit, a street type spelled out) |

Both have the columns of the built-in voter dataset plus two it does not have, taken from the real voter file's layout:

- `res_street_address`: house number and street, sometimes with an apartment or unit.
- `full_phone_number`: ten digits, no separators, present for about 43% of voters.

They also carry `cluster`, the ground truth: rows with the same value are the same voter. Delete that column to see
the app without accuracy figures.

## How to use them

1. On the first step choose **Use your own files**, then upload `voters_10k_A.csv` as Dataset A and type `ncid` as the
   ID column.
2. Pick one of:
   - **Find duplicates in Dataset A only**: finds the repeat registrations inside A.
   - **Upload a Dataset B to link with**: upload `voters_10k_B.csv` and type `ncid` as its ID column too.
3. On Configure, good starting rules are `dob` and a combined `first_name` + `last_name` rule. Compare
   `res_street_address` and `full_phone_number` along with the names and date of birth, and try the comparison type
   per field (`ExactMatch` or `LevenshteinAtThresholds` suit the address).

Regenerate them, or make a different size, with `python tools/make_upload_samples.py [folder] [rows]`.
