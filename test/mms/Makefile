all: poloidal radial parallel

radial: do_radial

.PHONY: do_radial

do_radial: .grids_radial
	python radial_mms.py

.grids_radial: radial_grid.py force
	python $<
	touch $@

poloidal: do_poloidal

.PHONY: do_poloidal

do_poloidal: .grids_poloidal
	python poloidal_mms.py

.grids_poloidal: poloidal_grid.py force
	python $<
	touch $@


parallel: do_parallel

.PHONY: do_parallel

do_parallel: .grids_parallel
	python parallel_mms.py

.grids_parallel: parallel_grid.py force
	python $<
	touch $@

laplace: do_laplace

.PHONY: do_laplace

do_laplace: .grids_laplace
	python laplace_mms.py

.grids_laplace: laplace_grid.py force
	python $<
	touch $@

force:
