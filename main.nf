#!/usr/bin/env nextflow
/*
========================================================================================
                          nf-rasqual
========================================================================================
                DeepSea Pipeline with nextflow.
                https://github.com/datngu/nf-deepfarm
                Author: Dat T Nguyen
                Contact: ndat<at>utexas.edu
----------------------------------------------------------------------------------------
*/





/*
 Define the default parameters
*/ 
params.genome          = "$baseDir/data/ref/genome.fa"
params.peaks           = "$baseDir/data/peak/*"

params.outdir          = "results"
params.trace_dir       = "trace_dir"

// running options
params.chrom           = 29 
params.val_chrom       = 21
params.test_chrom      = 25

params.window          = 200
params.seqlen          = 1000
//params.methods         = ['DanQ', 'DeepSEA', 'DeepFormer']
params.learning_rates  = [1e-3, 1e-4, 5e-4, 5e-5]



// pipeline options
params.generate_data  = true
params.train          = false
params.evaluate       = false
params.predict        = false


log.info """\
================================================================
                        nf-deepsea
================================================================
    genome              : $params.genome
    peaks               : $params.peaks
    outdir              : $params.outdir
    trace_dir           : $params.trace_dir


    chrom               : $params.chrom
    test_chrom          : $params.test_chrom
    val_chrom           : $params.val_chrom 
    window              : $params.window
    seqlen              : $params.seqlen


    generate_data       : $params.generate_data
    train               : $params.train
    evaluate            : $params.evaluate
    predict             : $params.predict


================================================================
"""

nextflow.enable.dsl=2



workflow {
    
    // hyper params:learning rates
    Learning_rate_ch = channel.from(params.learning_rates)
    //Hyper_params_ch = Method_ch.combine(Learning_rate_ch)
    //Method_ch = channel.from(params.methods)

    INDEX_genome(params.genome)
    BIN_genome(INDEX_genome.out)

    ch_peaks = channel.fromPath(params.peaks, checkIfExists: true)

    BED_mapping(BIN_genome.out, ch_peaks)
    
    LABEL_generating(BED_mapping.out.collect())
    
    //LABEL_generating.out.view()
    
    TFR_data_generating(LABEL_generating.out, BIN_genome.out, params.genome)
    
    DANQ_training_fw(TFR_data_generating.out, Learning_rate_ch)
    DEEPSEA_training_fw(TFR_data_generating.out, Learning_rate_ch)

    // Mymodel_training_fw(TFR_data_generating.out, Learning_rate_ch)
}






// preprocessing data



process INDEX_genome {
    container 'ndatth/deepsea:v0.0.0'
    publishDir "${params.outdir}/genome", mode: 'symlink', overwrite: true
    memory '8 GB'
    cpus 1

    input:
    path "genome.fa"

    output:
    path("genome.fa*")


    script:
    """
    samtools faidx genome.fa
    """
}


process BIN_genome {
    container 'ndatth/deepsea:v0.0.0'
    publishDir "${params.outdir}/bed_files", mode: 'symlink', overwrite: true
    memory '8 GB'
    cpus 1

    input:
    path genome

    output:
    path("genome_window.bed")


    script:
    """
    generate_coordinate_onebed.py --genome genome.fa.fai --out genome_window.bed --window $params.window --chrom $params.chrom
    """
}


process BED_mapping {
    container 'ndatth/deepsea:v0.0.0'
    publishDir "${params.outdir}/bed_files", mode: 'symlink', overwrite: true
    memory '8 GB'
    cpus 1

    input:
    path bed_path
    path peak

    output:
    path("positive_${peak}")


    script:
    """
    bedtools intersect -a $bed_path -b ${peak} -wo -f 0.50 > positive_${peak}
    """
}


process LABEL_generating {
    container 'ndatth/deepsea:v0.0.0'
    publishDir "${params.outdir}/peak_labels", mode: 'symlink', overwrite: true
    memory '32 GB'
    cpus 1

    input:
    path bed_path

    output:
    path("*.txt.gz")


    script:
    """
    generate_seq_labels.py --input positive_* --out '.'
    """
}



process TFR_data_generating {
    container 'ndatth/deepsea:v0.0.0'
    publishDir "${params.outdir}/tfr_data", mode: 'symlink', overwrite: true
    memory '32 GB'
    cpus 4
    label 'with_1gpu'
    

    input:
    path lab
    path bed
    path genome

    output:
    path("*.tfr")


    script:
    """
    for i in {1..${params.chrom}}
    do
        generate_tfr.py --label \${i}.txt.gz --bed $bed --genome $genome --pad_scale 5 --out \${i}
    done
    
    """
}


process DANQ_training_fw {
    container 'ndatth/deepsea:v0.0.0'
    publishDir "${params.outdir}/train", mode: 'symlink', overwrite: true
    memory '64 GB'
    cpus 8
    label 'with_1gpu'
    

    input:
    path tfr
    val lr

    output:
    path("DanQ_model*")


    script:
    """
    mv ${params.val_chrom}_fw.tfr ${params.val_chrom}_fw.val
    mv ${params.val_chrom}_rc.tfr ${params.val_chrom}_rc.val

    mv ${params.test_chrom}_fw.tfr ${params.test_chrom}_fw.test
    mv ${params.test_chrom}_rc.tfr ${params.test_chrom}_rc.test
    
    train_danq.py --train *_fw.tfr --val *_fw.val --out DanQ_model_${lr} --batch_size 1024 --lr ${lr}

    """
}


process DEEPSEA_training_fw {
    container 'ndatth/deepsea:v0.0.0'
    publishDir "${params.outdir}/train", mode: 'symlink', overwrite: true
    memory '64 GB'
    cpus 8
    label 'with_1gpu'
    

    input:
    path tfr
    val lr

    output:
    path("DanQ_model*")


    script:
    """
    mv ${params.val_chrom}_fw.tfr ${params.val_chrom}_fw.val
    mv ${params.val_chrom}_rc.tfr ${params.val_chrom}_rc.val

    mv ${params.test_chrom}_fw.tfr ${params.test_chrom}_fw.test
    mv ${params.test_chrom}_rc.tfr ${params.test_chrom}_rc.test
    
    train_danq.py --train *_fw.tfr --val *_fw.val --out DanQ_model_${lr} --batch_size 1024 --lr ${lr}

    """
}



// process DANQ_training_fw {
//     container 'ndatth/deepsea:v0.0.0'
//     publishDir "${params.outdir}/train_fw", mode: 'symlink', overwrite: true
//     memory '64 GB'
//     cpus 32
//     label 'with_1gpu'
    

//     input:
//     path tfr

//     output:
//     path("danq_model.*")


//     script:
//     """
//     mv ${params.val_chrom}_fw.tfr ${params.val_chrom}_fw.val
//     mv ${params.val_chrom}_rc.tfr ${params.val_chrom}_rc.val

//     mv ${params.test_chrom}_fw.tfr ${params.test_chrom}_fw.test
//     mv ${params.test_chrom}_rc.tfr ${params.test_chrom}_rc.test
    
//     train_danq.py --train *_fw.tfr --val *_fw.val --out danq_model --batch_size 1024

//     """
// }

