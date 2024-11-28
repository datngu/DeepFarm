#!/usr/bin/env nextflow

// include { foo } from './module/deltaSVM'


nextflow.enable.dsl=2


process ATAC_deltaSVM_slipt_bed {

    container 'ndatth/delta-svm:v0.0.0'
    publishDir "${params.outdir}/deltaSVM", mode: 'symlink', overwrite: true
    memory '16 GB'

    input:
    path atac_filtered
    
    output:
    path "atac_*.txt"

    script:
    """
    slipt_bed.R $atac_filtered $params.deltaSVM_folds atac_ $params.deltaSVM_length
    """
}


process ATAC_deltaSVM_gen_null_seqs {

    container 'ndatth/delta-svm:v0.0.0'
    publishDir "${params.outdir}/deltaSVM", mode: 'symlink', overwrite: true
    memory '64 GB'
    cpus 16

    input:
    path trf_bed
    path list_atac_bed
    
    output:
    path "*.fa"

    script:
    """
    for i in \$(seq 1 $params.deltaSVM_folds)
    do
        genNullSeqs_tfr.R bed=atac_\$i.txt trf=$trf_bed bsgenome="BSgenome.Salmo.Salar.Ensembl.106" xfold=1 out_prefix=\$i batchsize=20000 &
    done
    wait
    """
}


process ATAC_deltaSVM_train {

    container 'ndatth/delta-svm:v0.0.0'
    publishDir "${params.outdir}/deltaSVM", mode: 'symlink', overwrite: true
    memory '96 GB'
    cpus 16

    input:
    path list_atac_seqs
    
    output:
    path "*.model.txt"

    script:
    """
    for i in \$(seq 1 $params.deltaSVM_folds)
    do
        gkmtrain \${i}_posSet.fa \${i}_negSet.fa \${i} -l 10 -k 7 -d 3 &
    done
    wait
    """
}


process ATAC_deltaSVM_merge_models {

    container 'ndatth/delta-svm:v0.0.0'
    publishDir "${params.outdir}/deltaSVM", mode: 'symlink', overwrite: true
    memory '8 GB'
    cpus 1

    input:
    path list_models
    
    output:
    path "all_model_merged.txt"

    script:
    """
        merge_lsgkm_models.py *.model.txt > all_model_merged.txt
    """
}


process ATAC_deltaSVM_gen_10mers {

    container 'ndatth/delta-svm:v0.0.0'
    publishDir "${params.outdir}/deltaSVM", mode: 'symlink', overwrite: true
    memory '8 GB'
    cpus 1

    input:


    output:
    path "nr10mers.fa"

    script:
    """
        nrkmers.py 10 nr10mers.fa
    """
}



process ATAC_deltaSVM_score_10mers {

    container 'ndatth/delta-svm:v0.0.0'
    publishDir "${params.outdir}/deltaSVM", mode: 'symlink', overwrite: true
    memory '96 GB'
    cpus 16

    input:
    path models
    path "nr10mers.fa"

    output:
    path "*_nr10mer_scores.txt"

    script:
    """
        
        for i in \$(seq 1 $params.deltaSVM_folds)
        do
        gkmpredict "nr10mers.fa" "\${i}.model.txt" "\${i}_nr10mer_scores.txt" &
        done
        wait

    """
}


process ATAC_deltaSVM_average_weights {

    container 'ndatth/delta-svm:v0.0.0'
    publishDir "${params.outdir}/deltaSVM", mode: 'symlink', overwrite: true
    memory '8 GB'
    cpus 1

    input:
    path weights
    
    output:
    path "weight_merged.txt"

    script:
    """
        merge_lsgkm_weights.py --weight *_nr10mer_scores.txt --out "weight_merged.txt"
    """
}


process ATAC_deltaSVM_input_generator {

    container 'ndatth/delta-svm:v0.0.0'
    publishDir "${params.outdir}/deltaSVM", mode: 'symlink', overwrite: true
    memory '16 GB'
    cpus 1

    input:
    path genome
    path vcf

    output:
    path "deltaSVM_input*.fa"

    script:
    """
        deltaSVM_input_generator.py --genome $genome --vcf $vcf --out "deltaSVM_input"
    """
}

process ATAC_deltaSVM {

    container 'ndatth/delta-svm:v0.0.0'
    publishDir "${params.outdir}/deltaSVM_prediction", mode: 'copy', overwrite: true
    memory '16 GB'
    cpus 1

    input:
    path inputs
    path weights

    output:
    path "all_snp_deltaSVM.txt"

    script:
    """
        deltasvm.pl deltaSVM_input_ref.fa deltaSVM_input_alt.fa $weights "all_snp_deltaSVM.txt"
    """
}
