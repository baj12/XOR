python src/main.py data/raw/xor_data.csv --config config/config.yaml --log DEBUG 2>&1 |tee mylog.txt
python src/moveResults.py config.0

ls data/raw
xor_data.csv            xor_data.min0.5.csv     xor_data.min1.csv       xor_data.ratio_0.01.csv xor_data.shift0.5.csv   xor_data.spaced.csv

for fp in data/raw/* ;
do 
    #echo $fp
    filename="${fp##*/}"
    name_without_ext="${filename%.*}"
    for co in config/* ;
    do
        #echo $co
        coname="${co##*/}"
        co_without_ext="${coname%.*}"
        echo "python src/main.py $fp --config $co  2>&1 | tee plots/mylog.$name_without_ext.$co_without_ext.txt "
        echo python src/moveResults.py $name_without_ext.$co_without_ext
    done
done

conda activate xorProject-test
python src/main.py data/raw/xor_data.csv --config config/config.1.yaml  2>&1 | tee plots/mylog.xor_data.config.1.txt 
python src/moveResults.py xor_data.config.1
python src/main.py data/raw/xor_data.csv --config config/config.2.yaml  2>&1 | tee plots/mylog.xor_data.config.2.txt 
python src/moveResults.py xor_data.config.2
python src/main.py data/raw/xor_data.csv --config config/config.3.yaml  2>&1 | tee plots/mylog.xor_data.config.3.txt 
python src/moveResults.py xor_data.config.3
python src/main.py data/raw/xor_data.csv --config config/config.4.yaml  2>&1 | tee plots/mylog.xor_data.config.4.txt 
python src/moveResults.py xor_data.config.4
python src/main.py data/raw/xor_data.csv --config config/config.yaml  2>&1 | tee plots/mylog.xor_data.config.txt 
python src/moveResults.py xor_data.config
python src/main.py data/raw/xor_data.min0.5.csv --config config/config.1.yaml  2>&1 | tee plots/mylog.xor_data.min0.5.config.1.txt 
python src/moveResults.py xor_data.min0.5.config.1
python src/main.py data/raw/xor_data.min0.5.csv --config config/config.2.yaml  2>&1 | tee plots/mylog.xor_data.min0.5.config.2.txt 
python src/moveResults.py xor_data.min0.5.config.2
python src/main.py data/raw/xor_data.min0.5.csv --config config/config.3.yaml  2>&1 | tee plots/mylog.xor_data.min0.5.config.3.txt 
python src/moveResults.py xor_data.min0.5.config.3
python src/main.py data/raw/xor_data.min0.5.csv --config config/config.4.yaml  2>&1 | tee plots/mylog.xor_data.min0.5.config.4.txt 
python src/moveResults.py xor_data.min0.5.config.4
python src/main.py data/raw/xor_data.min0.5.csv --config config/config.yaml  2>&1 | tee plots/mylog.xor_data.min0.5.config.txt 
python src/moveResults.py xor_data.min0.5.config
python src/main.py data/raw/xor_data.min1.csv --config config/config.1.yaml  2>&1 | tee plots/mylog.xor_data.min1.config.1.txt 
python src/moveResults.py xor_data.min1.config.1
python src/main.py data/raw/xor_data.min1.csv --config config/config.2.yaml  2>&1 | tee plots/mylog.xor_data.min1.config.2.txt 
python src/moveResults.py xor_data.min1.config.2
python src/main.py data/raw/xor_data.min1.csv --config config/config.3.yaml  2>&1 | tee plots/mylog.xor_data.min1.config.3.txt 
python src/moveResults.py xor_data.min1.config.3
python src/main.py data/raw/xor_data.min1.csv --config config/config.4.yaml  2>&1 | tee plots/mylog.xor_data.min1.config.4.txt 
python src/moveResults.py xor_data.min1.config.4
python src/main.py data/raw/xor_data.min1.csv --config config/config.yaml  2>&1 | tee plots/mylog.xor_data.min1.config.txt 
python src/moveResults.py xor_data.min1.config
python src/main.py data/raw/xor_data.ratio_0.01.csv --config config/config.1.yaml  2>&1 | tee plots/mylog.xor_data.ratio_0.01.config.1.txt 
python src/moveResults.py xor_data.ratio_0.01.config.1
python src/main.py data/raw/xor_data.ratio_0.01.csv --config config/config.2.yaml  2>&1 | tee plots/mylog.xor_data.ratio_0.01.config.2.txt 
python src/moveResults.py xor_data.ratio_0.01.config.2
python src/main.py data/raw/xor_data.ratio_0.01.csv --config config/config.3.yaml  2>&1 | tee plots/mylog.xor_data.ratio_0.01.config.3.txt 
python src/moveResults.py xor_data.ratio_0.01.config.3
python src/main.py data/raw/xor_data.ratio_0.01.csv --config config/config.4.yaml  2>&1 | tee plots/mylog.xor_data.ratio_0.01.config.4.txt 
python src/moveResults.py xor_data.ratio_0.01.config.4
python src/main.py data/raw/xor_data.ratio_0.01.csv --config config/config.yaml  2>&1 | tee plots/mylog.xor_data.ratio_0.01.config.txt 
python src/moveResults.py xor_data.ratio_0.01.config
python src/main.py data/raw/xor_data.shift0.5.csv --config config/config.1.yaml  2>&1 | tee plots/mylog.xor_data.shift0.5.config.1.txt 
python src/moveResults.py xor_data.shift0.5.config.1
python src/main.py data/raw/xor_data.shift0.5.csv --config config/config.2.yaml  2>&1 | tee plots/mylog.xor_data.shift0.5.config.2.txt 
python src/moveResults.py xor_data.shift0.5.config.2
python src/main.py data/raw/xor_data.shift0.5.csv --config config/config.3.yaml  2>&1 | tee plots/mylog.xor_data.shift0.5.config.3.txt 
python src/moveResults.py xor_data.shift0.5.config.3
python src/main.py data/raw/xor_data.shift0.5.csv --config config/config.4.yaml  2>&1 | tee plots/mylog.xor_data.shift0.5.config.4.txt 
python src/moveResults.py xor_data.shift0.5.config.4
python src/main.py data/raw/xor_data.shift0.5.csv --config config/config.yaml  2>&1 | tee plots/mylog.xor_data.shift0.5.config.txt 
python src/moveResults.py xor_data.shift0.5.config
python src/main.py data/raw/xor_data.spaced.csv --config config/config.1.yaml  2>&1 | tee plots/mylog.xor_data.spaced.config.1.txt 
python src/moveResults.py xor_data.spaced.config.1
python src/main.py data/raw/xor_data.spaced.csv --config config/config.2.yaml  2>&1 | tee plots/mylog.xor_data.spaced.config.2.txt 
python src/moveResults.py xor_data.spaced.config.2
python src/main.py data/raw/xor_data.spaced.csv --config config/config.3.yaml  2>&1 | tee plots/mylog.xor_data.spaced.config.3.txt 
python src/moveResults.py xor_data.spaced.config.3
python src/main.py data/raw/xor_data.spaced.csv --config config/config.4.yaml  2>&1 | tee plots/mylog.xor_data.spaced.config.4.txt 
python src/moveResults.py xor_data.spaced.config.4
python src/main.py data/raw/xor_data.spaced.csv --config config/config.yaml  2>&1 | tee plots/mylog.xor_data.spaced.config.txt 
python src/moveResults.py xor_data.spaced.config


