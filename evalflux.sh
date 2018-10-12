python3 md_evalflux.py 0.52 80 300 0.5 &
python3 md_evalflux.py 0.52 80 300 1.0 &
wait
python3 md_evalflux.py 0.52 80 300 2.0
python3 md_evalflux.py 0.52 190 300 0.5 &
python3 md_evalflux.py 0.52 190 300 1.0 &
wait
python3 md_evalflux.py 0.52 190 300 2.0
python3 md_evalflux.py 0.52 190 300 4.0
python3 md_evalflux.py 0.52 190 300 8.0
python3 md_evalflux.py 0.52 80 300 4.0
python3 md_evalflux.py 0.52 80 300 8.0
python3 md_evalflux.py 0.52 80 300 16.0
python3 md_evalflux.py 0.52 300 300 0.5 &
python3 md_evalflux.py 0.52 300 300 1.0 &
python3 md_evalflux.py 0.52 300 300 4.0 &
wait
python3 md_evalflux.py 0.52 300 300 8.0

python3 multipleplots.py 80 DensTime &
python3 multipleplots.py 80 DensTimeGas &
wait
python3 multipleplots.py 300 DensTime &
python3 multipleplots.py 300 DensTimeGas &
wait

shutdown -h now

