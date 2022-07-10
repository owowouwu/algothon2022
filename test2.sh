for i in 0.1 0.5 1. 2. 5. 10.
do
	echo $i
	python grid.py $i &
done
