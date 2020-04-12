@echo off

set lrs = 0.0001 0.00001 0.000001 0.0000001 0.00000001
set lrfs = 0.1 0.5 0.9
set lrps = 1 2 10
set oms = 0.9 0.99
set bs = 5 8

FOR %%lr IN (%lrs%) DO
(
  FOR %%lrf IN (%lrfs%) DO
  (
    FOR %%lrp IN (%lrps%) DO
    (
      FOR %%om IN (%oms%) DO
      (
        FOR %%b IN (%bs%) DO
        (
          python train-modified.py -e 20 -l %lr% -lrf %lrf% -lrp %lrp% -om %om% -b %b% -s 1
        )
      )
    )   
  )
)