# 1. Generate test and train data
python data_gen_phase_plots.py

#2. Find optimal hyperparameters for each algorithm
# Set all to "cuda:0" if single GPU / "cpu" for cpu
python phase_plot_train_params.py --device 'cuda:0' --algo 'ISTA'
python phase_plot_train_params.py --device 'cuda:0' --algo 'FISTA'
python phase_plot_train_params.py --device 'cuda:0' --algo 'TF_ISTA'
python phase_plot_train_params.py --device 'cuda:0' --algo 'TF_FISTA'
python phase_plot_train_params.py --device 'cuda:0' --algo 'RTF_ISTA'
python phase_plot_train_params.py --device 'cuda:0' --algo 'RTF_FISTA'

python phase_plot_train_params.py --device 'cuda:1' --algo 'MCP'
python phase_plot_train_params.py --device 'cuda:1' --algo 'FMCP'
python phase_plot_train_params.py --device 'cuda:1' --algo 'TF_MCP'
python phase_plot_train_params.py --device 'cuda:1' --algo 'TF_FMCP'
python phase_plot_train_params.py --device 'cuda:1' --algo 'RTF_MCP'
python phase_plot_train_params.py --device 'cuda:1' --algo 'RTF_FMCP'

#3. Phase plots on test data
python phase_plots_test.py --algo 'ISTA' --device "cuda:0"
python phase_plots_test.py --algo 'FISTA'  --device "cuda:0"
python phase_plots_test.py --algo 'TF_ISTA'  --device "cuda:0"
python phase_plots_test.py --algo 'TF_FISTA' --device 'cuda:0'
python phase_plots_test.py --algo 'RTF_ISTA' --device 'cuda:0' 
python phase_plots_test.py --algo 'RTF_FISTA' --device 'cuda:0'

python phase_plots_test.py --algo 'MCP' --device 'cuda:1'
python phase_plots_test.py --algo 'FMCP' --device 'cuda:1'
python phase_plots_test.py --algo 'TF_MCP' --device 'cuda:1'
python phase_plots_test.py --algo 'TF_FMCP' --device 'cuda:1' 
python phase_plots_test.py --algo 'RTF_MCP'  --device 'cuda:1'
python phase_plots_test.py --algo 'RTF_FMCP'  --device 'cuda:1'

#4. Generate plots
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo ISTA --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo FISTA --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo TF_ISTA --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo TF_FISTA --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo RTF_ISTA --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo RTF_FISTA --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo MCP --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo FMCP --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo TF_MCP --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo TF_FMCP --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo RTF_MCP --device 'cuda:0'
python TF_MCP_SNR_vs_Iter.py --sparsity 20 --SNR 30 --algo RTF_FMCP --device 'cuda:0'
python TF_MCP_SNR_vs_Iter_generate_plot.py --sparsity 20 --SNR 30 --ylim -20 --iter 2000

#5. Generate Contours
python phase_plots_generate_countour.py --algo 'ISTA'
python phase_plots_generate_countour.py --algo 'FISTA'
python phase_plots_generate_countour.py --algo 'TF_ISTA'
python phase_plots_generate_countour.py --algo 'TF_FISTA'
python phase_plots_generate_countour.py --algo 'RTF_ISTA'
python phase_plots_generate_countour.py --algo 'RTF_FISTA'
python phase_plots_generate_countour.py --algo 'MCP'
python phase_plots_generate_countour.py --algo 'FMCP'
python phase_plots_generate_countour.py --algo 'TF_MCP'
python phase_plots_generate_countour.py --algo 'TF_FMCP'
python phase_plots_generate_countour.py --algo 'RTF_MCP'
python phase_plots_generate_countour.py --algo 'RTF_FMCP'