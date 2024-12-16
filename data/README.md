# Generated input data

## Values between -10 and 10

```
python src/data_generator.py
```

## values between -1 and 1
    (proper xor)
```
python src/data_generator.py --buffer 0.0 --min_val -1 --max_val 1 --ratio_classes 1  --noise_std 0.01 --output data/raw/xor_data.min1.csv
```

## values between -0.5 and 0.5
    std = 1
```
python src/data_generator.py --buffer 0.0 --min_val -0.5 --max_val 0.5 --ratio_classes 1  --noise_std 0.01 --output data/raw/xor_data.min0.5.csv
```

## shifted away from 0
```
python src/data_generator.py --buffer 0.0 --min_val 0.5 --max_val 1.5 --ratio_classes 1  --noise_std 0.01 --output data/raw/xor_data.shift0.5.csv
```

## different sizes of class
=> probably obvious that the smaller in extreme cases won't be able to be learnt.?
```
python src/data_generator.py --buffer 0.1 --min_val -1 --max_val 1 --ratio_classes 0.01  --noise_std 0.01 --output data/raw/xor_data.ratio_0.01.csv
```

## space between class
=> probably obvious that the smaller in extreme cases won't be able to be learnt.?
```
python src/data_generator.py --buffer 0.1 --min_val -1 --max_val 1 --noise_std 0.01 --output data/raw/xor_data.spaced.csv
```
