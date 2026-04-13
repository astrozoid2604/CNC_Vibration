# GenAI_vibration
Project repository for project development files

## Main Application
Two applications exist in this folder: `app.py` and `eda/example_eda.py`

1) Main Application (incl. anomaly detection)
```
python app.py
```
Note: there are feature flags to be enabled within the 'app.py'. For example, data conversion (from log to h5), data padding, anomaly detection and so on.

2) Data quality metrics for EDA
Details, refer to the 'Data Quality Metrics' section below.

## Failure Types and Generation Method:
```
> Ball_screw - Manually generated
> bartobarfailure - Actual production
> belt wheel - Manually generated
```

## Data Quality Metrics
Fidelity assesment of sythetic data generated is needed and demo file (`eval/example_quality_eval.py`) is included. Demo file accepts argument parsing and a sample to run the demo is shown below.  

Input argument to run at commandline. NOTE: Absolute path may be needed to parse in the locations for the *real data* and *test data*, respectively.
```
1) Demo mode    
python eval/example_quality_eval.py --tests all --demo

2) Single test:
python eval/example_quality_eval.py --real_data /home/user/demo/data/real --synthetic_data /home/user/demo/data/synthetic --tests single --single_test wasserstein

3) Multiple tests:
python eval/example_quality_eval.py --real_data /home/user/demo/data/real --synthetic_data /home/user/demo/data/synthetic --tests multiple --multiple_tests "wasserstein,ks"

4) ALL tests:
python eval/example_quality_eval.py --real_data /home/user/demo/data/real --synthetic_data /home/user/demo/data/synthetic --tests all
```

File references
```
eval/example_quality_eval.py # tutorial sample file
utils/quality_metrics.py # function
```

## Anomaly Detection Algo (off-the-shelf) results
Relevant results are stored in '/docs/past_results/' folder.

## Exploratory Data Analysis (EDA)
Relevant plots re stored in '/eda' folder.

## Environment reproducibility
Exported to `env/dgan1.yml`, for use in Ubuntu-based conda environment.

# Model 1: DoppelGANger (DGAN)

## Research Papers
In order to prepare yourself better with how vanilla GAN works and how DGAN specifically works, there are in total of 5 papers, all of which can be seen in directory `docs/DGAN_literature_review/DGAN_research_paper`. The main points of these 5 papers have been summarized in JAMESLIM_paper_summary.docx located in same directory mentioned just now.

## CNC Vibration Data EDA
Upon having theoretical understanding on DGAN as mentioned earlier, the next step is to get familiar with exploration and analysis of time-series vibration data from CNC machine whose raw data was provided during past internship attachment.
The step-by-step EDA processes are captured in this Jupyter Notebook located at `eda/cnc_tSNE_notebook.ipynb`. This step does not only refresh us on Python analysis, but it is also because the nature of CNC vibration data close resembles our vibration data of interest from RBCC (let's call it RBCC vibration data from here onwards).

## DGAN Evaluation Metrics
In order to gauge the performance of DGAN's synthetic data, we need to equip ourself with appropriate evaluation metrics. By doing this, we can reliably tune our hyperparameter settings. The prevalent metrics used can be seen in this Evaluation Metrics PPT located at `docs/DGAN_literature_review/Gen-AI_Evaluation_Metrics.pptx`.

## DGAN Training
Having done all the literature review steps (i.e. reading research papers & performing EDA practice on CNC vibration data), we are ready to train and tune our DGAN model on real-life RBCC vibration data. Detailed step-by-step instructions on how to get RBCC vibration data and how to run DGAN model can be checked from README.md located at `experiments/scripts/README.md`. 

Composition of `experiments/scripts/` directory:
- The training script is all consolidated in 1 Python file (except for evaluation metric computation which will be covered later), namely, `train_DGAN.py`. 
- `DGAN_debug.ipynb` acts as a sandbox in which all isolated debugging attempts or try-outs can be performed safely without affecting the integrity & functionatlity of our main training file as mentioned above.
- `df_minmax.csv` stores the minimum and maximum values of each vibration axes, namely, X axis, Y axis, and Z axis. During data wrangling, we have to perform data scaling in the range of [-1, 1] for each of aforelisted axes. After generating synthetic data, if we want to retrieve back the data in original scaling, we can make use of the minimum and maximum values stored in this CSV document.
- `quality_metrics.py` is a separate file from `train_DGAN.py` that computes various evaluation metrics.
- The DGAN model that we use is based on open-source [gretel-synthetic's timeseries_dgan library](https://github.com/gretelai/gretel-synthetics/tree/master/src/gretel_synthetics/timeseries_dgan). Since we want to add various additional capabilities, we need to edit the files in said library upon completing its installation (P.S.: the installation instructions of the library can be checked at `README.md` or at [gretel-synthetic's GITHUB](https://github.com/gretelai/gretel-synthetics)). As the library file is not part of this repository, several library files with names that begin with `dgan.bak.ver` are committed to [dgan/ Directory](https://github.boschdevcloud.com/ONH1SGP/rbcc_GenAI_vibration/tree/main/dgan). The details of each of the `dgan.bak.ver` files can be checked in [dgan/README.md](https://github.boschdevcloud.com/ONH1SGP/rbcc_GenAI_vibration/blob/main/dgan/README.md).
- `multispectral_loss_GAN.py` depicts a simplified vanilla GAN just to test the validity of spectral loss which is later integrated into `dgan.bak.ver6_with-spectral-loss.py`. 

## DGAN Experiments
The details of DGAN experiments settings and results can be seen in DGAN Experiment PPT located at `experiments/results/JAMESLIM_dgan_experiment_result.pptx`. 

As of 9-Jan-2024, the best proposed experiment settings are **Experiment 15** which is the result of cherry-picking the good traits of _Experiment 08 (for long epoch of 10000)_ and _Experiment 11c (for short epoch of 1500)_. In addition, there is the new inclusion of the _spectral_loss_ feature. The explanation of what each of the following parameter can be checked in `README.md`.
```
sample_len = 2634
epochs = 1500
attribute_num_layers = 16
attribute_num_units = 512
feature_num_layers = 16
feature_num_units = 512
generator_learning_rate = 1e-4
attribute_discriminator_learning_rate = 1e-4
apply_feature_scaling = False
discriminator_rounds = 3
generator_rounds = 1
epoch_patience = 1
delta_loss_patience = 1e-3
num_epoch_per_checkpoint = 2
oscillate_DLR = True
oscillate_GLR = False
KLdiv_loss_replacement = False
KLdiv_loss_addition = True

batch_size = 10
KLdiv_fft_loss_addition = False
spectral_loss = True
discriminator_learning_rate = 1e-3
```
Exp15 Interim Result can be checked at location `experiments/results/Exp15_interim_result.PNG`.

# Model 2: Anyway Conditional Tabular GAN (ACTGAN)

More details to be added in this section
