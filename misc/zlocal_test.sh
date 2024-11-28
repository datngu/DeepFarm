## export bin
export PATH=$PATH:$PWD/bin


cd /Users/datn/DATA_ANALYSES/OCR_prediction

bed_path=genome_window.bed


## slipt genome

generate_coordinate_onebed.py --genome /Users/datn/GENOMES/atlantic_salmon/Salmo_salar.Ssal_v3.1.dna.toplevel.fa.fai --out $bed_path --window 200 --chrom 29

## bed mapping

mkdir bed_mapping
for peak in data_downloaded/Salmon/*
do
    out_fn=$(basename $peak)
    bedtools intersect -a $bed_path -b ${peak} -wo -f 0.50 > bed_mapping/positive_${out_fn} &
done

mkdir bed_mapping_base
for peak in data_downloaded/Salmon/*
do
    out_fn=$(basename $peak)
    bedtools intersect -a $bed_path -b ${peak} -wo > bed_mapping_base/${out_fn}
done


generate_seq_labels.py --input bed_mapping/positive_* --out 'salmon_label'


debug_generate_tfr.py --label salmon_label/25.txt.gz --bed genome_window.bed --genome /Users/datn/GENOMES/atlantic_salmon/Salmo_salar.Ssal_v3.1.dna.toplevel.fa --pad_scale 5