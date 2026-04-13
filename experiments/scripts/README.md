# DoppelGANger (DGAN) Model Synthetic Time-Series Data Generation

##Directory Hierarchy (before any run of train_DGAN.py - i.e. fresh checkout)
<pre>
.
|-- rbcc_GenAI_vibration/          
    |-- __init__.py
    |-- eda/
    |-- utils/
    |-- results/
    |-- example_quality_eval.py
    |-- example_eda.py
    |-- README.md
    |-- app.py
    |-- dgan/
        |-- __init__.py
        |-- DGAN_debug.ipynb
        |-- README.md (this file)
        |-- df_minmax.csv
        |-- dgan.bak.ver1_with-tensorboard.py
        |-- dgan.bak.ver2_with-best-model.py
        |-- dgan.bak.ver3_with-checkpoint-patience.py
        |-- dgan.bak.ver4_with-varying-lr.py
        |-- dgan.bak.ver5_with-KLdivloss.py
        |-- dgan.bak.ver6_with-spectral-loss.py
        |-- multispectral_loss_GAN.py
        |-- quality_metrics.py
        |-- train_DGAN.py
</pre>

## DGAN Literature Review Files
01. All resource materials for DGAN can be found under JAMESLIM_dgan_literature_review/ directory at the root
02. Within said directory, there are 2 directories:
    - JAMESLIM_EDA_practice_on_JANBAISCH_timeseries_data/ 
      - Description: Contains EDA practice details on vibration timeseries datasets from CNC machine which were analyzed by JANBAISCH (i.e. intern before JAMESLIM)
    - JAMESLIM_GAN_research_paper/
      - Description: Contains all research papers that establishes understanding on vanilla GAN models, GAN implementation on timeseries, and eventually the DGAN model itself

## Step-by-Step Instruction to run train_DGAN.py
01. Login into server address 192.168.88.252 with username _mluser_intern_. Obtain the password from Kevin.
02. Clone this following GIT repository
    
    **CMD**: cd ~/Desktop; git clone https://github.boschdevcloud.com/ONH1SGP/rbcc_GenAI_vibration.git
03. Copy the post-processed RBCC data which are in h5 file format
    (Please note that all these following commands are assumed to be executed from root _(i.e. ./rbcc_GenAI_vibration/)_)
    
    **CMD**: mkdir data; cp -r /home/SharedFolder/rbcc_cutting/test/ data/
    Now, data/ directory is under rbcc_GenAI_vibration/
04. Install all dependencies required by Gretel-AI Synthetic's GIT repository

    **CMD**: git clone https://github.com/gretelai/gretel-synthetic/
    (Then follow the README.md there)
05. Activate virtual environment with all dependencies installed

    **CMD**: conda activate dgan1
06. Overwrite appropriate gretel-synthetic's dgan.py
      - List of filenames:
        - dgan/dgan.bak.ver1_with-tensorboard.py          => **Version 1**: Original gretel-synthetic's dgan.py + TensorBoard charting capability
        - dgan/dgan.bak.ver2_with-best-model.py           => **Version 2**: Version 1 + Best Model saving capability
        - dgan/dgan.bak.ver3_with-checkpoint-patience.py  => **Version 3**: Version 2 + Checkpoint saving capability + Patience termination capability
        - dgan/dgan.bak.ver4_with-varying-lr.py           => **Version 4**: Version 3 + Varying learning rate on generator & discriminator networks   
        - dgan/dgan.bak.ver5_with-KLdivloss.py            => **Version 5**: Version 4 + KL Divergence component in Discriminator's network
        - dgan/dgan.bak.ver6_with-spectral-loss.py        => **Version 6**: Version 5 + Spectral loss component in Generator's network
        - dgan/dgan.bak.ver7_with-corrected-loss.py       => **Version 7**: Version 6 + Corrected loss for KL divergence and Spectral loss computation in time domain instead of based on Discriminator's output
      - Next, overwrite default dgan.py file in gretel-synthetics package with corresponding modified dgan.py in your local repo accordingly. See example below
      
      **CMD**: cp /home/mluser_intern/Desktop/rbcc_git/dgan/dgan.bak.ver2_with-best-model.py /home/mluser_intern/miniconda3/envs/dgan1/lib/python3.9/site-packages/gretel_synthetics/timeseries_dgan/dgan.py
07. Run train_DGAN.py

    **CMD**: python3 dgan/train_DGAN.py <optional_argument1> <optional_argument2> ...

    **CMD Example for Exp15**: python3 dgan/train_DGAN.py --gpu_id=0 --batch_size=10 --spectral_loss=True --discriminator_learning_rate=1e-3

    List of optional arguments and their corresponding default values which are based on Experiment 13:
    
    | NAME OF OPTIONAL ARGUMENTS               | TYPE | DEFAULT VALUE | DESCRIPTION                                                                                                                                                     |
    | ---------------------------------------: | ---- | ------------: | -------------------------------------------------------------------------------------------------------------------------------------------                     |
    |                                 --gpu_id | int  |        1      | 'Select which GPU to run the DoppelGANger training on'                                                                                                          |
    |                --clean_data_from_scratch | int  |        0      | 'Flag for concatenating all h5 files of RBCC abnormal data + Standardizing data'                                                                                |
    |                        --data_imputation | int  |        0      | 'Flag for imputing data to ensure each example_id has same sequence length'                                                                                     |
    |                                  --train | int  |        1      | 'Flag for training DGAN model'                                                                                                                                  |
    |                     --generate_synthetic | int  |        1      | 'Flag for generating synthetic data based on a trained model'                                                                                                   |
    |                               --evaluate | int  |        1      | 'Flag for performing evaluation'                                                                                                                                |
    |                                 --exp_id | str  |   'Exp8'      | 'Experiment ID'                                                                                                                                                 |
    |                             --sample_len | int  |     2634      | 'Divisor of max_sequence_len. It is advised to be between 10 and 20. Must be a factor of 118530'                                                                |
    |                             --batch_size | int  |       18      | 'Recommended value in gretel-synthetics package is 1000, but will need to adjust based on computing power'                                                      |
    |                                 --epochs | int  |     1500      | 'For large dataset, it is appropriate to set epochs between 100 to 1000'                                                                                        |
    |                   --attribute_num_layers | int  |       16      | 'Number of Discriminator network\'s hidden layers'                                                                                                              |
    |                    --attribute_num_units | int  |      512      | 'Number of Discriminator network\'s hidden nodes'                                                                                                               |
    |                     --feature_num_layers | int  |       16      | 'Number of Generator     network\'s hidden layers'                                                                                                              |
    |                      --feature_num_units | int  |      512      | 'Number of Generator     network\'s hidden layers'                                                                                                              |
    |                --generator_learning_rate | float|     1e-4      | 'Adam learning rate for Generator     network'                                                                                                                  |
    |            --discriminator_learning_rate | float|     1e-4      | 'Adam learning rate for Discriminator network'                                                                                                                  |
    |  --attribute_discriminator_learning_rate | float|     1e-4      | 'Adam learning rate for Auxiliary Discriminator network'                                                                                                        |
    |                  --apply_feature_scaling | bool |    False      | 'Scale continuous variables inside the model. Can only set to False if inputs are already scaled to [0, 1] or [-1, 1] '                                         |
    |                   --discriminator_rounds | int  |        3      | 'Number of training steps per batch for Discriminator')                                                                                                         |
    |                       --generator_rounds | int  |        1      | 'Number of training steps per batch for Generator')                                                                                                             |
    |                         --epoch_patience | int  |        1      | 'Tolerance of maximum consecutive epoch with absolute delta loss values less than delta_loss_patience'                                                          |
    |                    --delta_loss_patience | float|     1e-3      | 'Minimum threshold for delta loss value that Generator or Discriminator networks must have within epoch window equals to epoch_patience'                        |
    |               --num_epoch_per_checkpoint | int  |        2      | 'Saving checkpoint states for every 2 epochs in training by default. Any abort event will avoid from training from epoch 0 all over again.'                     |
    |                          --oscillate_DLR | bool |     True      | 'Implement varying learning rate on Discriminator network'                                                                                                      |
    |                          --oscillate_GLR | bool |    False      | 'Implement varying learning rate on Generator     network'                                                                                                      |
    |                 --KLdiv_loss_replacement | bool |    False      | 'Replace loss_generated and loss_real with KL divergence loss in Discriminator\'s loss function within _train() function in gretel-synthetics\' dgan.py file')  |
    |                    --KLdiv_loss_addition | bool |     True      | 'Add KL divergence loss in time domain as the new components in in Discriminator\'s loss function within _train() function in gretel-synthetics\' dgan.py file')|
    |                --KLdiv_fft_loss_addition | bool |    False      | 'Add KL divergence loss in freq domain as the new components in in Discriminator\'s loss function within _train() function in gretel-synthetics\' dgan.py file')|
    |                   --spectral_loss_addition | bool |    False      | 'Add weighted MSE FFT loss across the frequency spectrum'                                                                    |

## Details about train_DGAN.py
01. Global Variables
    | NAME OF GLOBALS               | DEFAULT VALUE                                | REMARKS                                                                                                                                                                        |
    | ----------------------------: | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | 
    | failure_type                  | 'Ball_Screw'                                 |                                                                                                                                                                                | 
    | cwd_path                      | '/home/mluser_intern/Desktop/rbcc_git/dgan/' |                                                                                                                                                                                |
    | target_minmax_offset_filename | 'df_minmax.csv'                              |                                                                                                                                                                                | 
    | target_h5_real_filename       | 'clean_data.h5'                              |                                                                                                                                                                                |
    | target_h5_impt_filename       | 'clean_imputed_data.h5'                      |                                                                                                                                                                                |
    | target_model_filename         |                                              | The name convention is based on the argument values passed to the DGANConfig()                                                                                                 |
    | target_h5_synt_filename       | 'synt_data.h5'                               |                                                                                                                                                                                |
    | threshold_seq_len             | 150000                                       | This is done to avoid samples with unsually long sequences                                                                                                                     |
    | num_imputation_repeat         | 1000                                         | Since DataFrame with "Long" format requires the sequence length to be the same for all samples, last few sequences will be repeated to pad until max_sequence_len is satisfied |
02. There are _06 sections_ in this Python script:
    - **Long DataFrame Construction**
      - Long format means that features are the columns of the DataFrame and timestamps are the rows
      - All abnormal samples of chosen <b>failure_type</b> are concatenated along axis=0 (i.e. row direction) so the DataFrame becomes taller/longer
      - One additional column called 'example_id' is added which carries the sample sequence ID of said abnormal data. This is needed to ensure that the eventual DataFrame is not a single sample with very long sequence.
    - **Data Scaling**
      - All features (i.e. all 3 axes) are scaled to between [-1, 1], required for DGAN model.
    - **Data Imputation**
      - Since the sequence length of each sample is non-uniform, we need to impute the samples to match the sample with longest sequence
      - There are 2 imputation modes:
        - imputation_mode='replication1000' means that last 1000 timestamps are repeated until it covers maximum sequence length
        - imputation_mode='lastdata' means that last timestamp of a sample is repeated until it covers maximum sequence length
    - **DoppelGANger Training**
      - To train DGAN model based on Gretel-AI package (i.e. gretel-synthetics)
    - **Synthetic Data Generation**
      - To generate synthetic data based on learned DGAN model
    - **Performance Metric Evaluation**
      - To run computation of similarity score between ground-truth abnormal data and synthetic data using several proposed distance metrics.

## Algorithm for Termination Patience
Here is the code which we can see from dgan/dgan.bak.ver6_with-spectral-loss.py
```
            # START :: Terminating training session if delta_loss < self.delta_loss_patience within self.epoch_patience window. Model saving is taking care of in dgan/train_DGAN.py
            if epoch==0 or (checkpoint_found and epoch==start_epoch): #For 1st training epoch
                prev_generator_loss, prev_discriminator_loss = generator_loss, discriminator_loss
                prev_epoch = epoch
            else: #For subsequent training epoch
                if (np.abs(generator_loss-prev_generator_loss)>self.delta_loss_patience) or (np.abs(discriminator_loss-prev_discriminator_loss)>self.delta_loss_patience):
                    # If there delta loss of either Generator/Discriminator networks is greater than self.delta_loss_patience, prev_epoch pointer will be updated to current epoch
                    # Further, previous losses (prev_generator_loss & prev_discriminator_loss) are updated to current losses
                    prev_generator_loss, prev_discriminator_loss = generator_loss, discriminator_loss
                    prev_epoch = epoch

                if epoch-prev_epoch>self.epoch_patience:
                    # If there delta loss of either Generator/Discriminator networks is smaller than self.delta_loss_patience, prev_epoch pointer won't be updated
                    # As a result, if the difference between current epoch and prev_epoch pointer is greater than self.epoch_patience, terminate the training
                    break
            # FINISH:: Terminating training session if delta_loss < self.delta_loss_patience within self.epoch_patience window. Model saving is taking care of in dgan/train_DGAN.py
```

Explanation on the algorithm:
- In the code snippet above, generator_loss and discriminator_loss refers to the last loss value in each network on last training step. 
- In the 1st iteration of the epoch:
    - We set all "previous" variables with current states (i.e. the losses and the epoch).
        - For training from scratch, 1st iteration of epoch refers to epoch 0
        - For training from certain saved checkpoint states, 1st iteration refers to whichever epoch state is saved in said checkpoint states
- For subsequent training epoch, we will only update our "previous" variables only when at least 1 of the networks (i.e. Generator & Discriminator) has absolute loss delta larger than delta_loss_patience which is currently defaulted to 1e-3
    - The only scenario in which "previous" variables are not updated is when absolute loss delta of BOTH networks are smaller than delta_loss_patience
- When the lenght of current epoch and "previous" epoch is larger than epoch_patience (currently is defaulted to 1), we will terminate training by breaking out of epoch loop in DGAN.\_train() function.

Examples 01: epoch_patience 1, delta_loss_patience 1e-3
- Scenario A:
  |                                STATE VALUES                                           |
  |---------------------------------------------------------------------------------------|
  |Last checkpoint   at epoch  9, generator_loss        -10, discriminator_loss       -20 |
  |Continue training to epoch 10, generator_loss (-10+1e-5), discriminator_loss (-20-2e-3)|
    - At epoch 9, prev_epoch=9, prev_generator_loss=-10, prev_discriminator_loss=-20
    - At epoch 10, since absolute generator loss is 2e-3 which is larger than delta_loss_patience=1e-3, all "previous" variables will be updated to values in epoch 10
    - Conclusion: Left pointers are updated to new values and training will continue. It should be noted that "left pointer" here refers to the concept in sliding window algorithm

- Scenario B:
  |                                STATE VALUES                                           |
  |---------------------------------------------------------------------------------------|
  |Last checkpoint   at epoch  9, generator_loss        -10, discriminator_loss       -20 |
  |Continue training to epoch 10, generator_loss (-10+1e-5), discriminator_loss (-20-9e-4)|
  |Continue training to epoch 11, generator_loss (-10+1e-5), discriminator_loss (-20-1e-3)|
    - At epoch 9, prev_epoch=9, prev_generator_loss=-10, prev_discriminator_loss=-20
    - At epoch 10, since absolute loss delta of both networks are not greater than delta_loss_patience=1e-3, all "previous" variables remain the same
    - No training termination at epoch 10 since current epoch - prev_epoch = 1 which is not greater than epoch_patience=1
    - At epoch 11, since absolute loss delta of both networks are not greater than delta_loss_patience=1e-3, all "previous" variables remain the same
    - Training will be terminated at epoch 11 because current epoch - prev_epoch = 2 which is greater than epoch_patience=1. All checkpoint states at epoch 11 will be saved as well

Example 02: epoch_patience 3, delta_loss_patience 1e-3
- |                                STATE VALUES                                             |
  |-----------------------------------------------------------------------------------------|
  |Last checkpoint   at epoch 20, generator_loss        -10, discriminator_loss       -20   |
  |Continue training to epoch 21, generator_loss (-10+1e-5), discriminator_loss (-20-1e-4)  |
  |Continue training to epoch 22, generator_loss (-10+2e-5), discriminator_loss (-20-2e-4)  |
  |Continue training to epoch 23, generator_loss (-10+1e-4), discriminator_loss (-20-2e-3)  |
  |Continue training to epoch 24, generator_loss (-10+1e-4), discriminator_loss (-20-2.1e-3)|
  |Continue training to epoch 25, generator_loss (-10+2e-4), discriminator_loss (-20-2.2e-3)|
  |Continue training to epoch 26, generator_loss (-10+3e-4), discriminator_loss (-20-2.4e-3)|
  |Continue training to epoch 27, generator_loss (-10+4e-4), discriminator_loss (-20-2.5e-3)|
    - At epoch 20, prev_epoch=20, prev_generator_loss=-10, prev_discriminator_loss=-20
    - At epoch 21 until epoch 22, since absolute loss delta of both networks are not greater than delta_loss_patience=1e-3, all "previous" variables remain the same
    - At epoch 23, absolute loss delta of discriminator network is 2e-3 which is larger than delta_loss_patience and therefore all "previous" variables are updated as follows.
        - prev_epoch = 23, prev_generator_loss = (-10+1e-4), prev_discriminator_loss = (-20-2e-3)
    - At epoch 24 until epoch 26, there is no update on "previous" variables. Also, current epoch - prev_epoch is not larger than epoch_patience=3, so there is no training termination
    - At epoch 27, however, current epoch - prev_epoch = 4 which is larger than epoch_patience=3. Therefore, training is terminated at epoch 27. All checkpoint states at epoch 27 will be saved as well
