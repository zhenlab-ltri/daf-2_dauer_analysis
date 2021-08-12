# daf2-dauer-comparisons
Comparison analysis of daf-2 dauer connectome dataset with wildtype nondauer connectome datasets

## Required Software

- Python 3.8.2

## Usage

Install project dependencies
`pip3 install -r requirements.txt `

Make new folders `output`, `graphs` and `analysis`

1. Edit generate_tables.py with your values for:
   - ```connection_type```: cell-to-cell or neuron_pair (Make sure this matches data_organizer.py)
   - ```compare_contactome_with```: all_nondauer or L1-L3 (Make sure this matches data_organizer.py)
   - ```synapse_type```: count or size(Make sure this matches data_organizer.py)
   - ```zero_filter```: 10 (all data) or 'early_development' (filter to allow 1 zero in early development)
   - ```compare```: daf2-dauer or L3 (L3 was added for proof of concept purposes)
   - ```pvalue_cutoff```: Your desired pvalue threshold, 0.05 is default
   - ```fdr_correction```: True or False 

2. Run generate_tables.py

3. Repeat steps 1-2 until your required conditions are completed.

4. Make new folders in `analysis` folder named based on a description of your `zero_fliter` from above
   i.e. `all_connections`, `1_zero_in_early_development`, and `no_zeros` 

5. Copy and paste the outputs from steps 1-3 into their appropriate folders in analysis

6. Make new folder in `graphs` folder, name using your value entered for `compare` in the previous step

7. Within the `{your value for compare}` folder, make new folder called `nonparametric_bootstrapping`

8. Within `nonparametric_bootstrapping`folder, make new folder based on description of your `zero_filter`
   i.e. `all_connections`, `1_zero_in_early_development`, and `no_zeros` 


9. Edit connection_classification.py with your values for:
   - ```filter``` = '1_zero_in_early_development' or 'all_connections'
   - ```cutoff``` = pvalue threshold, 0.05 is default
   - ```find_shared_stable_pvalues``` = True or False (Whether you want to only look at connections with pvalues under the threshold for all 3 normalization methods)
   - ```compare``` = daf2-dauer or L3

   If you want to compare with L3:
    - Make `test` folder in `analysis` folder, 
    - Make new folders in `test` folder named based on a description of your `zero_fliter` from above
      i.e. `all_connections`, `1_zero_in_early_development`, and `no_zeros` 
    - Edit `job_dir` to: 
      
         `job_dir = f'./output/connection_lists/test/{filter}`
    - Edit `load_data` function to:
    
      `df_total = make_connection_key(f'./analysis/test/{filter}/count/total_changes.csv')`
      
      `df_input = make_connection_key(f'./analysis/test/{filter}/count/input_changes.csv')`
      
      `df_output = make_connection_key(f'./analysis/test/{filter}/count/output_changes.csv')`

10. Make new folder in `output` folder called `connection_lists`

11. Run `connection_classification.py`

12. Edit summary_plot.py with your values for:
    - ```compare``` = daf2-dauer or L3 (Make sure this is the same value as above)
    
13.  Run `summary_plot.py`

## Optional

- run `venn_diagram.py` to get a summary of the similarities between analyses of size vs. count 
