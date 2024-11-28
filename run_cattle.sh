#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1                
#SBATCH --job-name=DS-cattle   
#SBATCH --mem=4G                
#SBATCH --partition=gpu
#SBATCH --mail-user=nguyen.thanh.dat@nmbu.no
#SBATCH --mail-type=ALL


module load git/2.23.0-GCCcore-9.3.0-nodocs
module load Nextflow/21.03
module load singularity/rpm

git pull

# git clone https://github.com/datngu/nf-deepsea.git

genome='/mnt/users/ngda/genomes/cattle/Bos_taurus.ARS-UCD1.2.dna_sm.toplevel.fa'

export NXF_SINGULARITY_CACHEDIR=/mnt/users/ngda/sofware/singularity
export TOWER_ACCESS_TOKEN=eyJ0aWQiOiA3OTAxfS4xNGY5NTFmOWNiZmEwNjZhOGFkYzliZTg3MDc4YWI4ZTRiYTk4ZmI5



nextflow run main.nf -resume -w work_dir \
    --genome ${genome} \
    --chrom 29 \
    --val_chrom 21 \
    --test_chrom 25 \
    --window 200 \
    --seqlen 1000 \
    --peaks '/mnt/SCRATCH/ngda/data/Cattle/*.bed' \
    -with-tower