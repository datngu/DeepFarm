#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1                
#SBATCH --job-name=DS-ck   
#SBATCH --mem=4G                
#SBATCH --partition=gpu
#SBATCH --mail-user=nguyen.thanh.dat@nmbu.no
#SBATCH --mail-type=ALL


module load git/2.23.0-GCCcore-9.3.0-nodocs
module load Nextflow/21.03
module load singularity/rpm



genome='/mnt/users/ngda/genomes/chicken/Gallus_gallus.GRCg6a.dna.toplevel.fa'

export NXF_SINGULARITY_CACHEDIR=/mnt/users/ngda/sofware/singularity
export TOWER_ACCESS_TOKEN=eyJ0aWQiOiA3OTAxfS4xNGY5NTFmOWNiZmEwNjZhOGFkYzliZTg3MDc4YWI4ZTRiYTk4ZmI5

chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
random_chars=$(printf "%s" "${chars:RANDOM%${#chars}:1}${chars:RANDOM%${#chars}:1}")

nextflow run main.nf -resume -w work_dir \
    -name "deepfarm_chicken_${random_chars}" \
    --genome ${genome} \
    --chrom 30 \
    --val_chrom 21 \
    --test_chrom 25 \
    --window 200 \
    --seqlen 1000 \
    --peaks '/mnt/SCRATCH/ngda/data/Chicken/*.bed' \
    -with-tower