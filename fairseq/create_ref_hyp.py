
SYS1_OUTPUT = '/home/olab/itayitzhak/bpeplus/fairseq/checkpoints/de-en_opt2_medium_model/eval.txt'
SYS2_OUTPUT = '/home/olab/itayitzhak/bpeplus/fairseq/checkpoints/rerun_share_opt2/eval.txt'

def creat_ref_hyp():
    ref_load_file = open('/home/olab/itayitzhak/bpeplus/compare-mt/example/ref.med', 'w+', encoding='utf8')
    hyp_load_file = open('/home/olab/itayitzhak/bpeplus/compare-mt/example/hyp.med', 'w+', encoding='utf8')
    ref_no_load_file = open('/home/olab/itayitzhak/bpeplus/compare-mt/example/ref.large', 'w+', encoding='utf8')
    hyp_no_load_file = open('/home/olab/itayitzhak/bpeplus/compare-mt/example/hyp.large', 'w+', encoding='utf8')

    with open(SYS1_OUTPUT, 'r', encoding='utf8') as load_output_f:
        for line in load_output_f.readlines():
            if line[0] == 'T':
                ref_load_file.write(' '.join(line.split(' ')[1:]))
            elif line[0] == 'H':
                hyp_load_file.write(' '.join(line.split(' ')[1:]))

    with open(SYS2_OUTPUT, 'r', encoding='utf8') as no_load_output_f:
        for line in no_load_output_f.readlines():
            if line[0] == 'T':
                ref_no_load_file.write(' '.join(line.split(' ')[1:]))
            elif line[0] == 'H':
                hyp_no_load_file.write(' '.join(line.split(' ')[1:]))


    ref_load_file.close()
    hyp_load_file.close()
    ref_no_load_file.close()
    hyp_no_load_file.close()


if __name__ == "__main__":
    creat_ref_hyp()
