#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))
#define sqr(x) ((x)*(x))

inline void PyDict_SetStolenItem(PyObject *dict, const char *key, PyObject *object) {
	PyDict_SetItemString(dict, key, object);
	Py_DECREF(object);
}

inline double _getkernel( double h, double r2 ) {
	double coeff1, coeff2, coeff5;
	double hinv, hinv3, u;
	coeff1 = 8.0 / M_PI;
	coeff2 = coeff1 * 6.0;
	coeff5 = coeff1 * 2.0;

	hinv = 1.0 / h;
	hinv3 = hinv*hinv*hinv;
	u = sqrt(r2)*hinv;
	if (u < 0.5) {
		return hinv3 * ( coeff1 + coeff2*(u-1.0)*u*u );
	} else {
		return hinv3 * coeff5 * pow(1.0-u,3.0);
	}
}

PyObject* _calcGrid(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *rho, *value, *pyGrid;
	int npart, nx, ny, nz, cells;
	int dims[3];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_rho, *data_value;
	double *grid;
	int part;
	double px, py, pz, h, h2, m, r, v, cpx, cpy, cpz, r2;
	int x, y, z0, z1;
	int xmin, xmax, ymin, ymax, zmin, zmax, zmid;
	double cellsizex, cellsizey, cellsizez;
	time_t start;
	
	start = clock();

	if (!PyArg_ParseTuple( args, "O!O!O!O!O!iiidddddd:calcGrid( pos, hsml, mass, rho, value, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &rho, &PyArray_Type, &value, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz )) {
		return 0;
	}

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (hsml->nd != 1 || hsml->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "hsml has to be of dimension [n] and type double" );
		return 0;
	}

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (rho->nd != 1 || rho->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "rho has to be of dimension [n] and type double" );
		return 0;
	}

	if (value->nd != 1 || value->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]  || npart != rho->dimensions[0] || npart != value->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml, rho and value have to have the same size in the first dimension" );
		return 0;
	}

	dims[0] = nx;
	dims[1] = ny;
	dims[2] = nz;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 3, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny*nz;
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;
	cellsizez = bz / nz;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	data_rho = (double*)rho->data;
	data_value = (double*)value->data;

	for (part=0; part<npart; part++) {
		px = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		py = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		pz = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		r = *data_rho;
		data_rho = (double*)((char*)data_rho + rho->strides[0]);

		v = *data_value;
		data_value = (double*)((char*)data_value + value->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );
		zmin = max( floor( (pz - h - cz + 0.5*bz) / cellsizez ), 0 );
		zmax = min( ceil( (pz + h - cz + 0.5*bz) / cellsizez ), nz-1 );

		zmid = floor( 0.5 * (zmin+zmax) + 0.5 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && zmin < nz && zmax >= 0) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					for (z0=zmid; z0>=zmin; z0--) {
						cpz = -0.5*bz + bz*(z0+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						grid[(x*ny + y)*nz + z0] += _getkernel( h, r2 ) * m * v / r;
					}

					for (z1=zmid+1; z1<=zmax; z1++) {
						cpz = -0.5*bz + bz*(z1+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						grid[(x*ny + y)*nz + z1] += _getkernel( h, r2 ) * m * v / r;
					}
				}
			}	
		}
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _calcSlice(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *rho, *value, *pyGrid;
	int npart, nx, ny, cells;
	int dims[2];
	double bx, by, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_rho, *data_value;
	double *grid;
	int part;
	double px, py, pz, h, m, r, v, cpx, cpy, r2, h2;
	double p[3];
	int x, y;
	int xmin, xmax, ymin, ymax, axis0, axis1;
	double cellsizex, cellsizey;
	time_t start;
	
	start = clock();

	axis0 = 0;
	axis1 = 1;
	if (!PyArg_ParseTuple( args, "O!O!O!O!O!iiddddd|ii:calcSlice( pos, hsml, mass, rho, value, nx, ny, boxx, boxy, centerx, centery, centerz, [axis0, axis1] )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &rho, &PyArray_Type, &value, &nx, &ny, &bx, &by, &cx, &cy, &cz, &axis0, &axis1 )) {
		return 0;
	}

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (hsml->nd != 1 || hsml->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "hsml has to be of dimension [n] and type double" );
		return 0;
	}

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (rho->nd != 1 || rho->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "rho has to be of dimension [n] and type double" );
		return 0;
	}

	if (value->nd != 1 || value->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]  || npart != rho->dimensions[0] || npart != value->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml, mass, rho and value have to have the same size in the first dimension" );
		return 0;
	}
	dims[0] = nx;
	dims[1] = ny;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny;
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	data_rho = (double*)rho->data;
	data_value = (double*)value->data;

	for (part=0; part<npart; part++) {
		p[0] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[1] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[2] = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		px = p[ axis0 ];
		py = p[ axis1 ];
		pz = p[ 3 - axis0 - axis1 ];
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		r = *data_rho;
		data_rho = (double*)((char*)data_rho + rho->strides[0]);

		v = *data_value;
		data_value = (double*)((char*)data_value + value->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && abs(pz-cz) < h) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					r2 = sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cz);
					if (r2 > h2) continue;
					grid[x*ny + y] += _getkernel( h, r2 ) * m * v / r;
				}
			}	
		}
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _calcGridMassWeight(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *value, *pyGridMass, *pyGridValue;
	int npart, nx, ny, nz, cells;
	int dims[3];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_value;
	double *gridmass, *gridvalue, *massend, *massiter, *valueiter;
	int part;
	double px, py, pz, h, h2, m, v, cpx, cpy, cpz, r2, dmass;
	int x, y, z0, z1;
	int xmin, xmax, ymin, ymax, zmin, zmax, zmid;
	double cellsizex, cellsizey, cellsizez;
	time_t start;
	
	start = clock();

	if (!PyArg_ParseTuple( args, "O!O!O!O!O!iiidddddd:calcGridMassWeight( pos, hsml, mass, value, massgrid, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &value, &PyArray_Type, &pyGridMass, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz )) {
		return 0;
	}

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (hsml->nd != 1 || hsml->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "hsml has to be of dimension [n] and type double" );
		return 0;
	}

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (value->nd != 1 || value->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0] || npart != value->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and value have to have the same size in the first dimension" );
		return 0;
	}

	if (pyGridMass->nd != 3 || pyGridMass->dimensions[0] != nx || pyGridMass->dimensions[1] != ny || pyGridMass->dimensions[2] != nz) {
		PyErr_SetString( PyExc_ValueError, "massgrid has to have 3 dimensions: [nx,ny,nz]" );
		return 0;
	}

	dims[0] = nx;
	dims[1] = ny;
	dims[2] = nz;
	cells = nx*ny*nz;
	
	gridmass = (double*)pyGridMass->data;
	
	pyGridValue = (PyArrayObject *)PyArray_FromDims( 3, dims, PyArray_DOUBLE );
	gridvalue = (double*)pyGridValue->data;
	memset( gridvalue, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;
	cellsizez = bz / nz;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	data_value = (double*)value->data;

	for (part=0; part<npart; part++) {
		px = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		py = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		pz = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		v = *data_value;
		data_value = (double*)((char*)data_value + value->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );
		zmin = max( floor( (pz - h - cz + 0.5*bz) / cellsizez ), 0 );
		zmax = min( ceil( (pz + h - cz + 0.5*bz) / cellsizez ), nz-1 );

		zmid = floor( 0.5 * (zmin+zmax) + 0.5 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && zmin < nz && zmax >= 0) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					for (z0=zmid; z0>=zmin; z0--) {
						cpz = -0.5*bz + bz*(z0+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						
						dmass = _getkernel( h, r2 ) * m;
						gridvalue[(x*ny + y)*nz + z0] += dmass * v;
					}

					for (z1=zmid+1; z1<=zmax; z1++) {
						cpz = -0.5*bz + bz*(z1+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						
						dmass = _getkernel( h, r2 ) * m;
						gridvalue[(x*ny + y)*nz + z1] += dmass * v;
					}
				}
			}	
		}
	}
	
	massend = &gridmass[ cells ];
	for (massiter = gridmass, valueiter = gridvalue; massiter != massend; massiter++, valueiter++) {
		if (*massiter > 0)
			*valueiter /= *massiter;
	}
	
	return PyArray_Return( pyGridValue );
}

PyObject* _calcDensGrid(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *pyGrid;
	int npart, nx, ny, nz, cells;
	int dims[3];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass;
	double *grid;
	int part;
	double px, py, pz, h, h2, v, cpx, cpy, cpz, r2;
	int x, y, z0, z1;
	int xmin, xmax, ymin, ymax, zmin, zmax, zmid;
	double cellsizex, cellsizey, cellsizez;
	time_t start;
	
	start = clock();
	
	if (!PyArg_ParseTuple( args, "O!O!O!iiidddddd:calcDensGrid( pos, hsml, mass, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz )) {
		return 0;
	}

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (hsml->nd != 1 || hsml->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "hsml has to be of dimension [n] and type double" );
		return 0;
	}

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and mass have to have the same size in the first dimension" );
		return 0;
	}

	dims[0] = nx;
	dims[1] = ny;
	dims[2] = nz;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 3, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny*nz;
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;
	cellsizez = bz / nz;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	
	for (part=0; part<npart; part++) {
		px = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		py = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		pz = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		v = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );
		zmin = max( floor( (pz - h - cz + 0.5*bz) / cellsizez ), 0 );
		zmax = min( ceil( (pz + h - cz + 0.5*bz) / cellsizez ), nz-1 );

		zmid = floor( 0.5 * (zmin+zmax) + 0.5 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && zmin < nz && zmax >= 0) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					for (z0=zmid; z0>=zmin; z0--) {
						cpz = -0.5*bz + bz*(z0+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						grid[(x*ny + y)*nz + z0] += _getkernel( h, r2 ) * v;
					}

					for (z1=zmid+1; z1<=zmax; z1++) {
						cpz = -0.5*bz + bz*(z1+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						grid[(x*ny + y)*nz + z1] += _getkernel( h, r2 ) * v;
					}
				}
			}	
		}
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _calcDensSlice(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *pyGrid;
	int npart, nx, ny, cells;
	int dims[2];
	double bx, by, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass;
	double *grid;
	int part;
	double px, py, pz, h, v, cpx, cpy, r2, h2;
	double p[3];
	int x, y;
	int xmin, xmax, ymin, ymax, axis0, axis1;
	double cellsizex, cellsizey;
	time_t start;
	
	start = clock();

	axis0 = 0;
	axis1 = 1;
	if (!PyArg_ParseTuple( args, "O!O!O!iiddddd|ii:calcDensSlice( pos, hsml, mass, nx, ny, boxx, boxy, centerx, centery, centerz, [axis0, axis1] )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &nx, &ny, &bx, &by, &cx, &cy, &cz, &axis0, &axis1 )) {
		return 0;
	}

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (hsml->nd != 1 || hsml->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "hsml has to be of dimension [n] and type double" );
		return 0;
	}

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and mass have to have the same size in the first dimension" );
		return 0;
	}

	dims[0] = nx;
	dims[1] = ny;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny;
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;

	for (part=0; part<npart; part++) {
		p[0] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[1] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[2] = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		px = p[ axis0 ];
		py = p[ axis1 ];
		pz = p[ 3 - axis0 - axis1 ];
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		v = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && abs(pz-cz) < h) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					r2 = sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cz);
					if (r2 > h2) continue;
					grid[x*ny + y] += _getkernel( h, r2 ) * v;
				}
			}	
		}
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _calcCylinderAverage(PyObject *self, PyObject *args) {
	PyArrayObject *pyGrid, *pyNewgrid;
	int dims[2], cells;
	double *newgrid, *count;
	int x, y, z, nx, ny, nz, nr, r;

	if (!PyArg_ParseTuple( args, "O!:calcCylinderAverage( grid )", &PyArray_Type, &pyGrid )) {
		return 0;
	}

	if (pyGrid->nd != 3 || pyGrid->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "grid has to be of dimensions [nx,ny,nz] and type double" );
		return 0;
	}

	nx = pyGrid->dimensions[0];
	ny = pyGrid->dimensions[1];
	nz = pyGrid->dimensions[2];
	nr = min( ny, nz );
	
	dims[0] = nx;
	dims[1] = nr;
	cells = nx*nr;
	pyNewgrid = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	newgrid = (double*)pyNewgrid->data;
	memset( newgrid, 0, cells*sizeof(double) );

	count = (double*)malloc( cells*sizeof(double) );
	memset( count, 0, cells*sizeof(double) );

	for (x=0; x<nx; x++) for (y=0; y<ny; y++) for (z=0; z<nz; z++) {
		r = floor( sqrt( sqr(y-ny/2.0+0.5) + sqr(z-nz/2.0+0.5) ) );
		if (r >= nr/2) continue;

		newgrid[x*nr+r+nr/2] += *(double*)( pyGrid->data + pyGrid->strides[0]*x + pyGrid->strides[1]*y + pyGrid->strides[2]*z );
		count[x*nr+r+nr/2] += 1;
	}
	
	for (x=0; x<nx; x++) for (r=0; r<nr/2; r++) {
		if (count[x*nr+r+nr/2] > 0) {
			newgrid[x*nr+r+nr/2] /= count[x*nr+r+nr/2];
			newgrid[x*nr-r+nr/2-1] = newgrid[x*nr+r+nr/2];
		}
	}

	free( count );
	return PyArray_Return( pyNewgrid );
}

PyObject* _calcRadialProfile(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *data, *pyProfile;
	int npart, nshells, mode;
	int dims[2];
	int *count;
	double cx, cy, cz, dr;
	double *data_pos, *data_data;
	double *profile;
	int part, shell;
	double px, py, pz, d, rr, v;
	time_t start;
	
	start = clock();

	mode = 1;
	nshells = 200;
	dr = 0;
	cx = cy = cz = 0;
	if (!PyArg_ParseTuple( args, "O!O!|iidddd:calcRadialProfile( pos, data, mode, nshells, dr, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &data, &mode, &nshells, &dr, &cx, &cy, &cz )) {
		return 0;
	}

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (data->nd != 1 || data->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "data has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != data->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos and data have to have the same size in the first dimension" );
		return 0;
	}
	dims[0] = 2;
	dims[1] = nshells;
	pyProfile = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	profile = (double*)pyProfile->data;
	memset( profile, 0, 2*nshells*sizeof(double) );

	count = (int*)malloc( nshells*sizeof(int) );
	memset( count, 0, nshells*sizeof(int) );

	if (!dr) {
		data_pos = (double*)pos->data;
		for (part=0; part<npart; part++) {
			px = *data_pos;
			data_pos = (double*)((char*)data_pos + pos->strides[1]);
			py = *data_pos;
			data_pos = (double*)((char*)data_pos + pos->strides[1]);
			pz = *data_pos;
			data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);

			rr = sqrt( sqr(px-cx) + sqr(py-cy) + sqr(pz-cz) );
			if (rr > dr)
				dr = rr;
		}
		dr /= nshells;
		printf( "dr set to %g\n", dr );
	}

	data_pos = (double*)pos->data;
	data_data = (double*)data->data;

	for (part=0; part<npart; part++) {
		px = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		py = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		pz = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);

		d = *data_data;
		data_data = (double*)((char*)data_data + data->strides[0]);

		rr = sqrt( sqr(px-cx) + sqr(py-cy) + sqr(pz-cz) );
		shell = floor( rr / dr );

		if (shell < nshells) {
			profile[ shell ] += d;
			count[ shell ] += 1;
		}
	}

	for (shell=0; shell<nshells; shell++) {
		profile[ nshells + shell ] = dr * (shell + 0.5);
	}

	switch (mode) {
		// sum
		case 0:
			break;
		// density
		case 1:
			for (shell=0; shell<nshells; shell++) {
				v = 4.0 / 3.0 * M_PI * dr*dr*dr * ( ((double)shell+1.)*((double)shell+1.)*((double)shell+1.) - (double)shell*(double)shell*(double)shell );
				profile[shell] /= v;
			}
			break;
		// average
		case 2:
			for (shell=0; shell<nshells; shell++) if (count[shell] > 0) profile[shell] /= count[shell];
			break;
	}

	free( count );

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyProfile );
}

PyObject* _calcAbundGrid(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *abund, *pyGrid;
	int npart, nx, ny, nz, cells, nspecies;
	int dims[4];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_abund;
	double *grid;
	int part, species;
	double *xnuc;
	double px, py, pz, h, h2, m, cpx, cpy, cpz, r2, kk;
	int x, y, z0, z1;
	int xmin, xmax, ymin, ymax, zmin, zmax, zmid;
	double cellsizex, cellsizey, cellsizez;
	time_t start;
	
	start = clock();

	if (!PyArg_ParseTuple( args, "O!O!O!O!iiidddddd:calcAbundGrid( pos, hsml, mass, abund, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &abund, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz )) {
		return 0;
	}

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (hsml->nd != 1 || hsml->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "hsml has to be of dimension [n] and type double" );
		return 0;
	}

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (abund->nd != 2 || abund->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "abund has to be of dimension [n,nspecies] and type double" );
		return 0;
	}
	
	nspecies = abund->dimensions[1];

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]  || npart != abund->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and abund have to have the same size in the first dimension" );
		return 0;
	}
	
	xnuc = (double*)malloc( nspecies * sizeof( double ) );

	dims[0] = nx;
	dims[1] = ny;
	dims[2] = nz;
	dims[3] = nspecies+1;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 4, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny*nz*(nspecies+1);
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;
	cellsizez = bz / nz;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	data_abund = (double*)abund->data;

	for (part=0; part<npart; part++) {
		px = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		py = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		pz = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		for (species = 0; species < nspecies; species++ ) {
			xnuc[species] = *data_abund;
			data_abund = (double*)((char*)data_abund + abund->strides[1]);
		}
		data_abund = (double*)((char*)data_abund - nspecies*abund->strides[1] + abund->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );
		zmin = max( floor( (pz - h - cz + 0.5*bz) / cellsizez ), 0 );
		zmax = min( ceil( (pz + h - cz + 0.5*bz) / cellsizez ), nz-1 );

		zmid = floor( 0.5 * (zmin+zmax) + 0.5 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && zmin < nz && zmax >= 0) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					for (z0=zmid; z0>=zmin; z0--) {
						cpz = -0.5*bz + bz*(z0+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						
						kk = _getkernel( h, r2 ) * m;
						for (species = 0; species < nspecies; species++ )
							grid[((x*ny + y)*nz + z0)*(nspecies+1) + species] += kk * xnuc[species];
						grid[((x*ny + y)*nz + z0)*(nspecies+1) + nspecies] += kk;
					}

					for (z1=zmid+1; z1<=zmax; z1++) {
						cpz = -0.5*bz + bz*(z1+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						
						kk = _getkernel( h, r2 ) * m;
						for (species = 0; species < nspecies; species++ )
							grid[((x*ny + y)*nz + z1)*(nspecies+1) + species] += kk * xnuc[species];
						grid[((x*ny + y)*nz + z1)*(nspecies+1) + nspecies] += kk;
					}
				}
			}	
		}
	}
	
	free( xnuc );

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _calcAbundSphere(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *abund, *pyGrid;
	int npart, nradius, ntheta, nphi, cells, nspecies;
	int dims[4];
	double radius, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_abund;
	double *grid;
	int part, species;
	double *xnuc;
	double px, py, pz, h, h2, m, r, dr, cpx, cpy, cpz, r2, kk;
	double vr, vtheta, vphi;
	int ir, itheta, iphi;
	int minradius, maxradius;
	time_t start;
	
	start = clock();

	if (!PyArg_ParseTuple( args, "O!O!O!O!iiidddd:calcAbundSphere( pos, hsml, mass, abund, nradius, ntheta, nphi, radius, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &abund, &nradius, &ntheta, &nphi, &radius, &cx, &cy, &cz )) {
		return 0;
	}

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (hsml->nd != 1 || hsml->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "hsml has to be of dimension [n] and type double" );
		return 0;
	}

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (abund->nd != 2 || abund->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "abund has to be of dimension [n,nspecies] and type double" );
		return 0;
	}
	
	nspecies = abund->dimensions[1];

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]  || npart != abund->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and abund have to have the same size in the first dimension" );
		return 0;
	}
	
	xnuc = (double*)malloc( nspecies * sizeof( double ) );

	dims[0] = nradius;
	dims[1] = ntheta;
	dims[2] = nphi;
	dims[3] = nspecies+1;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 4, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nradius*ntheta*nphi*(nspecies+1);
	memset( grid, 0, cells*sizeof(double) );

	dr = radius / nradius;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	data_abund = (double*)abund->data;
		
	for (part=0; part<npart; part++) {
		px = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		py = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		pz = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		for (species = 0; species < nspecies; species++ ) {
			xnuc[species] = *data_abund;
			data_abund = (double*)((char*)data_abund + abund->strides[1]);
		}
		data_abund = (double*)((char*)data_abund - nspecies*abund->strides[1] + abund->strides[0]);

		r = sqrt( px*px + py*py + pz*pz );
		minradius = max( 0, floor( (r-h-0.5) / dr ) );
		maxradius = min( nradius-1, floor( (r+h+0.5) / dr ) );

		for (ir=minradius; ir<=maxradius; ir++)
		for (itheta=0; itheta<ntheta; itheta++)
		for (iphi=0; iphi<nphi; iphi++) {
	        	vr = radius * (ir+0.5) / nradius;
			vtheta = M_PI * (itheta+0.5) / ntheta;
			vphi = 2. * M_PI * (iphi+0.5) / nphi;
			cpx = vr * sin( vtheta ) * cos( vphi );
			cpy = vr * sin( vtheta ) * sin( vphi );
			cpz = vr * cos( vtheta );

			r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
			if (r2 > h2) continue;
						
			kk = _getkernel( h, r2 ) * m;
			for (species = 0; species < nspecies; species++ ) {
				grid[((ir*ntheta + itheta)*nphi + iphi)*(nspecies+1) + species] += kk * xnuc[species];
			}
			grid[((ir*ntheta + itheta)*nphi + iphi)*(nspecies+1) + nspecies] += kk;
		}
	}
	
	free( xnuc );

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

static PyMethodDef calcGridmethods[] = {
	{ "calcGrid", _calcGrid, METH_VARARGS, "" },
	{ "calcSlice", _calcSlice, METH_VARARGS, "" },
	{ "calcDensGrid", _calcDensGrid, METH_VARARGS, "" },
	{ "calcDensSlice", _calcDensSlice, METH_VARARGS, "" },
	{ "calcGridMassWeight", _calcGridMassWeight, METH_VARARGS, "" },
	{ "calcCylinderAverage", _calcCylinderAverage, METH_VARARGS, "" },
	{ "calcRadialProfile", _calcRadialProfile, METH_VARARGS, "" },
	{ "calcAbundGrid", _calcAbundGrid, METH_VARARGS, "" },
	{ "calcAbundSphere", _calcAbundSphere, METH_VARARGS, "" },
	{ NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initcalcGrid(void)
{
	Py_InitModule( "calcGrid", calcGridmethods );
	import_array();
}
