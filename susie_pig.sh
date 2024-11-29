#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1                
#SBATCH --job-name=susie-pig   
#SBATCH --mem=4G                
#SBATCH --partition=gpu
#SBATCH --mail-user=nguyen.thanh.dat@nmbu.no
#SBATCH --mail-type=ALL


module load Nextflow/24.04.2
module load singularity/rpm


#genome='/mnt/users/ngda/genomes/pig/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa'

export NXF_SINGULARITY_CACHEDIR=/mnt/users/ngda/sofware/singularity
export TOWER_ACCESS_TOKEN=eyJ0aWQiOiA3OTAxfS4xNGY5NTFmOWNiZmEwNjZhOGFkYzliZTg3MDc4YWI4ZTRiYTk4ZmI5


SPEC="pig"
PICK_MODEL="DanQ_model_0.0005.h5"
mkdir -p selected_model
cp results/train/${PICK_MODEL} selected_model/model.h5

genome='/mnt/users/ngda/genomes/pig/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa'
col_file="$PWD/data/col_files/${SPEC}_colnames.txt"

chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
random_chars=$(printf "%s" "${chars:RANDOM%${#chars}:1}${chars:RANDOM%${#chars}:1}")

nextflow run main_susie.nf -resume \
    -w "work_dir_susie" \
    -name "susie_pig_keras_${random_chars}" \
    --col_file "$col_file" \
    --model "$PWD/selected_model/model.h5" \
    --vcfs "$PWD/data/susie_pig_gvf/*.txt" \
    --genome "$genome" \
    -with-tower