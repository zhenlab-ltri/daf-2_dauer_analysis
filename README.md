# daf2-dauer-comparisons
Comparison analysis of daf-2 dauer connectome dataset with wildtype nondauer connectome datasets

## Required Software

- Python 3.8.2

## Usage

Install project dependencies
`pip3 install -r requirements.txt `

Make new folders `output`, `graphs` and `analysis`

1. Edit data_organizer.py wity your values for:
   - ```connection_type```: cell-to-cell or neuron_pair
   - ```compare_contactome_with```: all_nondauer or L1-L3
   - ```normalize_by```: input, output or entire_dataset
   - ```synapse_type```: count or size

2. Run `data_organizer.py`

3. Edit generate_tables.py with your values for:
   - ```connection_type```: cell-to-cell or neuron_pair (Make sure this matches data_organizer.py)
   - ```compare_contactome_with```: all_nondauer or L1-L3 (Make sure this matches data_organizer.py)
   - ```synapse_type```: count or size(Make sure this matches data_organizer.py)
   - ```zero_filter```: 10 (All data) or 'early_development' (filter to allow 1 zero in early development)
   - ```compare```: daf2-dauer or L3 (L3 was added for proof of concept purposes)
   - ```pvalue_cutoff```: Your desired pvalue threshold, 0.05 is default
   - ```fdr_correction```: True or False 

4. Run generate_tables.py

5. Repeat steps 1-4 until your required conditions are completed.

6. Make new folder in `graphs` folder, name using your value entered for `compare` in the previous step

7. Within the `{your value for compare}` folder, make new folder called `nonparametric_bootstrapping`


8. Edit connection_classification.py with your values for:
   - ```filter``` = '1_zero_in_early_development' or 'all_connections'
   - ```cutoff``` = pvalue threshold, 0.05 is default
   - ```find_shared_stable_pvalues``` = True or False (Whether you want to only look at connections with pvalues under the threshold for all 3 normalization methods)
   - ```compare``` = daf2-dauer or L3

9. Make new folder in `output` folder called `connection_lists`

10. Run `connection_classification.py`

11. Run `summary_plot.py`
