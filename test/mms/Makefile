all: laplace delp poloidal radial parallel

radial: do_radial

.PHONY: do_radial

do_radial: .grids_radial
	python3 radial_mms.py

FORCE=force
.grids_radial: radial_grid.py $(FORCE)
	python3 $<
	touch $@

poloidal: do_poloidal

.PHONY: do_poloidal

do_poloidal: .grids_poloidal
	python3 poloidal_mms.py

.grids_poloidal: poloidal_grid.py $(FORCE)
	python3 $<
	touch $@


parallel: do_parallel

.PHONY: do_parallel

do_parallel: .grids_parallel
	python3 parallel_mms.py

.grids_parallel: parallel_grid.py $(FORCE)
	python3 $<
	touch $@

laplace: do_laplace

.PHONY: do_laplace

do_laplace: .grids_laplace
	python3 laplace_mms.py

.grids_laplace: laplace_grid.py $(FORCE)
	python3 $<
	touch $@

delp: .grids_delp
	python3 delp_mms.py

.grids_delp: delp_grid.py $(FORCE)
	python3 $<
	touch $@

force:
