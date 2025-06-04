#!/usr/bin/env bash

action() {

    # set python path in current directory
    this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export PYTHONPATH="${this_dir}:${PYTHONPATH}"

    export LAW_HOME="${this_dir}/.law"
    export LAW_CONFIG_FILE="${this_dir}/law.cfg"

    # set input and output directories
    CONFIG_FILE="${this_dir}/.config"
    # Function to read the output directory from the config file
    read_config() {
        if [[ -f $CONFIG_FILE ]]; then
            source $CONFIG_FILE
        else
            export OUTPUT_DIR=""
            export DATA_DIR=""
        fi
    }

    # Function to write the input and output directory to the config file
    write_config() {
        echo "export OUTPUT_DIR=\"$OUTPUT_DIR\"" > $CONFIG_FILE
        echo "export DATA_DIR=\"$DATA_DIR\"" >> $CONFIG_FILE
        echo "Configuration saved to $CONFIG_FILE"
    }

    # Prompt user for input if OUTPUT_DIR and DATA_DIR are not set
    prompt_user() {
        read -p "Enter the output directory: " user_input1
        read -p "Enter the input directory: " user_input2
        if [[ -d $user_input1 && -d $user_input2 ]]; then
            export OUTPUT_DIR="$user_input1"
            export DATA_DIR="$user_input2"
        else
            echo "Invalid directories. Please try again."
            prompt_user
        fi
        write_config
    }

    read_config

    if [[ -z $OUTPUT_DIR ]]; then
        echo "No output directory configured."
        prompt_user
    fi

    echo "Using output directory: $OUTPUT_DIR"
    echo "Using input directory: $DATA_DIR"

    # Load necessary modules and activate conda environment
    # module load python
    # conda activate /pscratch/sd/m/mukyu/SBI_RANODE/env/sbi_ranode_env
    # module load miniconda CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
    # conda activate /gpfs/gibbs/pi/demers/runze/conda/envs/sbi_ranode_env

}

action