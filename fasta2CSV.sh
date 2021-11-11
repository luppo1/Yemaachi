sed '/^>/s/ .*//' 16s_sequences.fasta > 16s_sequences_1.fasta
sed -e 's/\(^>.*$\)/#\1#/' 16s_sequences_1.fasta | tr -d "\r" | tr -d "\n" | sed -e 's/$/#/' | tr "#" "\n" | sed -e '/^$/d' > 16s_sequences_2.fasta
~/albert/fastx_bin/bin/./fasta_formatter -t -i 16s_sequences_2.fasta -o 16s_sequences_final.csv

