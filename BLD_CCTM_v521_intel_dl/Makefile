
 EXEC = CCTM_v521.exe

 FC = mpiifort
 CC = icc

 LIB = /lustre/home/acct-esehazenet/hazenet-pg3/CMAQv5.2.1AIr/lib/x86_64/intel
 PLIB = /lustre/home/acct-esehazenet/hazenet-pg3/.conda/envs/lcpy/
 include_path = -I $(LIB)/ioapi/modules/fixed_src \
                -I $(LIB)/ioapi/include_files \
                -I $(LIB)/mpi -I.

 WARN = 
 FSTD = -O3 -fno-alias -mp1 -fp-model source 
 DBG  = -O0 -g -check bounds -check uninit -fpe0 -fno-alias -ftrapuv -traceback

 ifeq "$(DEBUG)" "TRUE" 
     f_FLAGS   = -fixed -132 $(DBG) $(include_path)
     f90_FLAGS = -free $(DBG) $(include_path)
 else
     f_FLAGS   = -fixed -132 $(FSTD) $(include_path)
     f90_FLAGS = -free $(FSTD) $(include_path)
 endif

 F_FLAGS   = $(f_FLAGS)
 F90_FLAGS = $(f90_FLAGS)
 C_FLAGS   = -O2 -DFLDMN -I$(LIB)/mpi/include -I.

 LINKER     = $(FC)
 LINK_FLAGS = -qopenmp

 REPOROOT = /lustre/home/acct-esehazenet/hazenet-pg3/5.2.1/CCTM/src

 CPP = $(FC)
 CPP_FLAGS = \
  -Dparallel \
  -DSUBST_BARRIER=SE_BARRIER \
  -DSUBST_GLOBAL_MAX=SE_GLOBAL_MAX \
  -DSUBST_GLOBAL_MIN=SE_GLOBAL_MIN \
  -DSUBST_GLOBAL_MIN_DATA=SE_GLOBAL_MIN_DATA \
  -DSUBST_GLOBAL_TO_LOCAL_COORD=SE_GLOBAL_TO_LOCAL_COORD \
  -DSUBST_GLOBAL_SUM=SE_GLOBAL_SUM \
  -DSUBST_GLOBAL_LOGICAL=SE_GLOBAL_LOGICAL \
  -DSUBST_LOOP_INDEX=SE_LOOP_INDEX \
  -DSUBST_SUBGRID_INDEX=SE_SUBGRID_INDEX \
  -DSUBST_HI_LO_BND_PE=SE_HI_LO_BND_PE \
  -DSUBST_SUM_CHK=SE_SUM_CHK \
  -DSUBST_INIT_ARRAY=SE_INIT_ARRAY \
  -DSUBST_COMM=SE_COMM \
  -DSUBST_MY_REGION=SE_MY_REGION \
  -DSUBST_SLICE=SE_SLICE \
  -DSUBST_GATHER=SE_GATHER \
  -DSUBST_DATA_COPY=SE_DATA_COPY \
  -DSUBST_IN_SYN=SE_IN_SYN

 IOAPI  = -L$(LIB)/ioapi/lib -lioapi
 NETCDF = -L$(LIB)/netcdf/lib -lnetcdf -lnetcdff
 PYTHON = -L$(PLIB)/lib -lpython3.6m
 LIBRARIES = $(IOAPI) $(NETCDF) $(PYTHON)

 BASE_INC = .
 MPI_INC  = $(LIB)/mpi/include

 INCLUDES =  \
  -DSUBST_PE_COMM=\"$(BASE_INC)/PE_COMM.EXT\" \
  -DSUBST_CONST=\"$(BASE_INC)/CONST.EXT\" \
  -DSUBST_FILES_ID=\"$(BASE_INC)/FILES_CTM.EXT\" \
  -DSUBST_EMISPRM=\"$(BASE_INC)/EMISPRM.EXT\" \
  -DSUBST_MPI=\"$(MPI_INC)/mpif.h\"


GLOBAL_MODULES = \
  UTILIO_DEFN.o \
  HGRD_DEFN.o \
  VGRD_DEFN.o \
  GRID_CONF.o \
  CGRID_SPCS.o \
  PA_DEFN.o \
  PCGRID_DEFN.o \
  AVG_CONC.o \
  DEPVVARS.o \
  BIDI_MOD.o \
  LSM_MOD.o \
  ASX_DATA_MOD.o \
  HGSIM.o \
  MOSAIC_MOD.o \
  RXNS_DATA_MODULE.o \
  BIOG_EMIS.o \
  BEIS_DEFN.o \
  LUS_DEFN.o \
  LTNG_DEFN.o \
  AEROMET_DATA.o \
  AERO_DATA.o \
  UDTYPES.o \
  STK_EMIS.o \
  STK_PRMS.o \
  PTMAP.o \
  PTBILIN.o \
  SSEMIS.o \
  bmatvec.o \
  PTMET.o \
  DEPV_DEFN.o \
  STD_CONC.o \
  CSQY_DATA.o \
  CLOUD_OPTICS.o \
  PHOT_MET_DATA.o \
  DUST_EMIS.o \
  RXNS_FUNC_MODULE.o \
  PRECURSOR_DATA.o \
  MGEMIS.o \
  PT3D_DEFN.o \
  SOA_DEFN.o \
  AERO_EMIS.o \
  AEROSOL_CHEMISTRY.o \
  PAGRD_DEFN.o \
  EMIS_DEFN.o \
  WVEL_DEFN.o

STENEX = \
  se_bndy_copy_info_ext.o \
  se_pe_info_ext.o \
  se_comm_info_ext.o \
  se_data_send_module.o \
  se_domain_info_ext.o \
  se_internal_util_module.o \
  se_data_recv_module.o \
  se_disp_info_ext.o \
  se_subgrid_info_ext.o \
  se_global_sum_module.o \
  se_global_max_module.o \
  se_global_min_module.o \
  se_slice_module.o \
  se_init_module.o \
  se_bndy_copy_module.o \
  se_comm_module.o \
  se_reconfig_grid_info_ext.o \
  se_reconfig_grid_module.o \
  se_gather_module.o \
  se_data_copy_module.o \
  se_term_module.o \
  se_twoway_comm_module.o \
  se_util_module.o \
  se_modules.o \
  swap_sandia.o \
  swap_sandia_routines.o

PARIO = \
  alloc_data_mod.o \
  pinterpb_mod.o \
  parutilio.o \
  piomaps_mod.o \
  boundary.o \
  get_write_map.o \
  growbuf.o \
  gtndxhdv.o \
  interpol.o \
  pinterpb.o \
  pio_init.o \
  pm3err.o \
  pm3exit.o \
  pm3warn.o \
  pshut3.o \
  ptrwrite3.o \
  pwrgrdd.o \
  pwrite3.o \
  readbndy.o \
  subdmap.o \
  wrsubdmap.o

PAR = \
  distr_env.o \
  mpcomm_init.o

DRIVER = \
  advstep.o \
  driver.o \
  hveloc.o \
  sciproc.o \
  wr_aconc.o \
  wr_cgrid.o \
  wr_conc.o \
  wr_init.o

INIT = \
  diffmsg.o \
  flcheck.o \
  grdcheck.o \
  initscen.o \
  load_cgrid.o \
  opaconc.o \
  opconc.o

COUPLE = \
  couplewrf.o

HADV = \
  advbc_map.o \
  hadvyppm.o \
  hcontvel.o \
  hppm.o \
  rdbcon.o \
  x_ppm.o \
  x_yamo.o \
  y_ppm.o \
  y_yamo.o \
  zfdbc.o

VADV = \
  vppm.o \
  zadvppmwrf.o

HDIFF = \
  deform.o \
  hcdiff3d.o \
  hdiff.o \
  rho_j.o

VDIFF = \
  VDIFF_DIAG.o \
  VDIFF_MAP.o \
  SEDIMENTATION.o \
  aero_sedv.o \
  conv_cgrid.o \
  eddyx.o \
  matrix1.o \
  opddep.o \
  opddep_fst.o \
  opddep_mos.o \
  rddepv.o \
  tri.o \
  vdiffacmx.o \
  vdiffproc.o

DEPV = \
  ABFLUX_MOD.o \
  cgrid_depv.o \
  gas_depv_map.o \
  m3dry.o \
  opdepv_diag.o \
  opdepv_fst.o \
  opdepv_mos.o

EMIS = \
  cropcal.o \
  opemis.o \
  tfabove.o \
  tfbelow.o

BIOG = \
  beis3.o \
  checkmem.o \
  chkgrid.o \
  czangle.o \
  getfline.o \
  getparb.o \
  hrno.o \
  parsline.o \
  tmpbeis.o \
  wrdaymsg.o

PLRISE = \
  delta_zs.o \
  fire_plmris.o \
  openlayout.o \
  oppt3d_diag.o \
  plmris.o \
  plsprd.o \
  preplm.o \
  ungridb2.o \
  write3_distr.o

PHOT = \
  complex_number_module.o \
  twoway_rrtmg_aero_optics.o \
  AERO_PHOTDATA.o \
  PHOT_MOD.o \
  SEAS_STRAT_O3_MIN.o \
  PHOTOLYSIS_ALBEDO.o \
  o3totcol.o \
  opphot.o \
  phot.o

GAS = \
  hrdata_mod.o \
  hrdriver.o \
  chem_py.o\
  forpy_mod.o\
  resnet.o\
  hrg1.o \
  hrg2.o \
  hrg3.o \
  hrg4.o \
  hrinit.o \
  hrprodloss.o \
  hrrates.o \
  hrsolver.o

AERO = \
  AOD_DEFN.o \
  PMDIAG_DATA.o \
  aero_depv.o \
  aero_driver.o \
  aero_subs.o \
  coags.o \
  getpar.o \
  isocom.o \
  isofwd.o \
  isorev.o \
  opapmdiag.o \
  opavis.o \
  oppmdiag.o \
  opvis.o

CLOUD = \
  AQ_DATA.o \
  acmcld.o \
  aqchem.o \
  aq_map.o \
  cldproc_acm.o \
  convcld_acm.o \
  getalpha.o \
  hlconst.o \
  indexn.o \
  opwdep.o \
  rescld.o \
  scavwdep.o
  
PROCAN = \
  PA_GLOBAL.o \
  PA_IPRDEF.o \
  PA_VARS.o \
  PA_PARSE.o \
  PA_IPRVARS.o \
  pa_compmech.o \
  pa_datagen.o \
  pa_errcheck.o \
  pa_getcoef.o \
  pa_getcycle.o \
  pa_getdesc.o \
  pa_getfamily.o \
  pa_getiprout.o \
  pa_getirrout.o \
  pa_getrxns.o \
  pa_getrxnsum.o \
  pa_init.o \
  pa_irr_ctl.o \
  pa_irr.o \
  pa_mkhdr.o \
  pa_molcloss.o \
  pa_molcprod.o \
  pa_output.o \
  pa_read.o \
  pa_report.o \
  pa_setup_ipr.o \
  pa_setup_irr.o \
  pa_update.o \
  pa_wrtpadefn.o

UTIL = \
  cksummer.o \
  findex.o \
  get_envlist.o \
  lstepf.o \
  setup_logdev.o \
  subhdomain.o \
  subhfile.o

OBJS = \
  $(STENEX) \
  $(PARIO) \
  $(GLOBAL_MODULES) \
  $(PAR) \
  $(DRIVER) \
  $(INIT) \
  $(COUPLE) \
  $(HADV) \
  $(VADV) \
  $(HDIFF) \
  $(VDIFF) \
  $(DEPV) \
  $(EMIS) \
  $(BIOG) \
  $(PLRISE) \
  $(PHOT) \
  $(GAS) \
  $(AERO) \
  $(CLOUD) \
  $(PROCAN) \
  $(UTIL)

.SUFFIXES: .F .f .c .F90 .f90

$(EXEC): $(OBJS)
	$(LINKER) $(LINK_FLAGS) $(OBJS) $(LIBRARIES) -o $@

.F.o:
	$(FC) -c $(F_FLAGS) $(CPP_FLAGS) $(INCLUDES) $<

.f.o:
	$(FC) -c $(F_FLAGS) $<

.F90.o:
	$(FC) -c $(F90_FLAGS) $(CPP_FLAGS) $(INCLUDES) $<

.f90.o:
	$(FC) -c $(F90_FLAGS) $<

.c.o:
	$(CC) -c $(C_FLAGS) $<

clean:
	rm -f $(OBJS) *.mod


# dependencies

se_bndy_copy_module.o:	se_bndy_copy_info_ext.o se_internal_util_module.o \
		se_data_send_module.o se_data_recv_module.o se_pe_info_ext.o \
		se_domain_info_ext.o
se_comm_module.o:	se_comm_info_ext.o se_internal_util_module.o \
		se_data_send_module.o se_data_recv_module.o se_pe_info_ext.o \
		se_domain_info_ext.o
se_data_copy_module.o:	se_subgrid_info_ext.o se_pe_info_ext.o \
		se_data_send_module.o se_data_recv_module.o se_domain_info_ext.o
se_data_recv_module.o:	se_pe_info_ext.o
se_data_send_module.o:	se_pe_info_ext.o
se_gather_module.o:	se_pe_info_ext.o se_slice_module.o se_global_sum_module.o
se_global_max_module.o:	se_pe_info_ext.o
se_global_min_module.o:	se_pe_info_ext.o
se_global_sum_module.o:	se_pe_info_ext.o
se_init_module.o:	se_pe_info_ext.o se_domain_info_ext.o se_comm_info_ext.o \
		se_disp_info_ext.o se_bndy_copy_info_ext.o se_internal_util_module.o
se_internal_util_module.o:	se_pe_info_ext.o
se_modules.o:	se_init_module.o se_term_module.o se_util_module.o \
		se_comm_module.o se_slice_module.o se_data_copy_module.o se_gather_module.o \
		se_reconfig_grid_module.o se_bndy_copy_module.o se_global_max_module.o \
		se_global_min_module.o se_global_sum_module.o se_twoway_comm_module.o
se_reconfig_grid_module.o:	se_reconfig_grid_info_ext.o se_pe_info_ext.o \
		se_data_send_module.o se_data_recv_module.o se_domain_info_ext.o
se_slice_module.o:	se_pe_info_ext.o
se_term_module.o:	se_pe_info_ext.o se_domain_info_ext.o se_reconfig_grid_info_ext.o \
		se_subgrid_info_ext.o
se_twoway_comm_module.o:	se_comm_info_ext.o
se_util_module.o:	se_comm_info_ext.o se_domain_info_ext.o se_pe_info_ext.o \
		se_subgrid_info_ext.o se_internal_util_module.o
swap_sandia_routines.o:	swap_sandia.o
get_write_map.o:	piomaps_mod.o
growbuf.o:	pinterpb_mod.o
gtndxhdv.o:	pinterpb_mod.o
parutilio.o:	pinterpb_mod.o
pinterpb.o:	pinterpb_mod.o parutilio.o
pio_init.o:	piomaps_mod.o
pshut3.o:	piomaps_mod.o
ptrwrite3.o:	piomaps_mod.o
pwrgrdd.o:	piomaps_mod.o alloc_data_mod.o
pwrite3.o:	piomaps_mod.o
readbndy.o:	piomaps_mod.o pinterpb_mod.o
mpcomm_init.o:	UTILIO_DEFN.o
advstep.o:	GRID_CONF.o UTILIO_DEFN.o se_modules.o $(BASE_INC)/CONST.EXT \
		$(BASE_INC)/FILES_CTM.EXT
AVG_CONC.o:	VGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o
driver.o:	PCGRID_DEFN.o RXNS_DATA_MODULE.o CGRID_SPCS.o STD_CONC.o forpy_mod.o \
		AVG_CONC.o WVEL_DEFN.o PA_DEFN.o PAGRD_DEFN.o UTILIO_DEFN.o \
		se_modules.o $(BASE_INC)/FILES_CTM.EXT
hveloc.o:	GRID_CONF.o UTILIO_DEFN.o se_modules.o $(BASE_INC)/PE_COMM.EXT \
		$(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
sciproc.o:	UTILIO_DEFN.o se_modules.o PA_DEFN.o
STD_CONC.o:	HGRD_DEFN.o VGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o
wr_aconc.o:	GRID_CONF.o AVG_CONC.o UTILIO_DEFN.o se_modules.o \
		$(BASE_INC)/FILES_CTM.EXT
wr_cgrid.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o se_modules.o \
		$(BASE_INC)/FILES_CTM.EXT
wr_conc.o:	GRID_CONF.o STD_CONC.o WVEL_DEFN.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
wr_init.o:	GRID_CONF.o WVEL_DEFN.o STD_CONC.o UTILIO_DEFN.o se_modules.o \
		$(BASE_INC)/FILES_CTM.EXT
WVEL_DEFN.o:	GRID_CONF.o UTILIO_DEFN.o
GRID_CONF.o:	HGRD_DEFN.o VGRD_DEFN.o
HGRD_DEFN.o:	UTILIO_DEFN.o
PAGRD_DEFN.o:	GRID_CONF.o se_modules.o PA_DEFN.o
PCGRID_DEFN.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o
VGRD_DEFN.o:	UTILIO_DEFN.o
diffmsg.o:	UTILIO_DEFN.o
flcheck.o:	UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
grdcheck.o:	GRID_CONF.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
initscen.o:	HGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o se_modules.o \
		$(BASE_INC)/FILES_CTM.EXT
load_cgrid.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o RXNS_DATA_MODULE.o \
		AERO_DATA.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
opaconc.o:	GRID_CONF.o AVG_CONC.o UTILIO_DEFN.o se_modules.o $(BASE_INC)/FILES_CTM.EXT
opconc.o:	GRID_CONF.o WVEL_DEFN.o STD_CONC.o UTILIO_DEFN.o se_modules.o \
		$(BASE_INC)/FILES_CTM.EXT
couplewrf.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
advbc_map.o:	CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
hadvyppm.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o se_modules.o \
		$(BASE_INC)/PE_COMM.EXT $(BASE_INC)/FILES_CTM.EXT
hcontvel.o:	GRID_CONF.o UTILIO_DEFN.o se_modules.o $(BASE_INC)/PE_COMM.EXT \
		$(BASE_INC)/FILES_CTM.EXT
hppm.o:	HGRD_DEFN.o UTILIO_DEFN.o se_modules.o
rdbcon.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o RXNS_DATA_MODULE.o \
		AERO_DATA.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
x_ppm.o:	HGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o se_modules.o $(BASE_INC)/PE_COMM.EXT \
		$(BASE_INC)/CONST.EXT
x_yamo.o:	HGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o se_modules.o \
		$(BASE_INC)/PE_COMM.EXT $(BASE_INC)/CONST.EXT
y_ppm.o:	HGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o se_modules.o $(BASE_INC)/PE_COMM.EXT \
		$(BASE_INC)/CONST.EXT
y_yamo.o:	HGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o se_modules.o \
		$(BASE_INC)/PE_COMM.EXT $(BASE_INC)/CONST.EXT
vppm.o:	CGRID_SPCS.o UTILIO_DEFN.o
zadvppmwrf.o:	GRID_CONF.o CGRID_SPCS.o WVEL_DEFN.o se_modules.o \
		UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
deform.o:	GRID_CONF.o UTILIO_DEFN.o se_modules.o $(BASE_INC)/PE_COMM.EXT \
		$(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
hcdiff3d.o:	GRID_CONF.o UTILIO_DEFN.o se_modules.o $(BASE_INC)/PE_COMM.EXT \
		$(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
hdiff.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o se_modules.o $(BASE_INC)/PE_COMM.EXT \
		$(BASE_INC)/CONST.EXT
rho_j.o:	GRID_CONF.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
aero_sedv.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o ASX_DATA_MOD.o \
		AERO_DATA.o AEROMET_DATA.o
ASX_DATA_MOD.o:	GRID_CONF.o LSM_MOD.o DEPVVARS.o UTILIO_DEFN.o \
		se_modules.o $(BASE_INC)/PE_COMM.EXT $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
conv_cgrid.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/CONST.EXT \
		$(BASE_INC)/FILES_CTM.EXT
eddyx.o:	GRID_CONF.o ASX_DATA_MOD.o UTILIO_DEFN.o
matrix1.o:	VGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o
opddep.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
opddep_fst.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o LSM_MOD.o \
		$(BASE_INC)/FILES_CTM.EXT
opddep_mos.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o LSM_MOD.o \
		$(BASE_INC)/FILES_CTM.EXT
rddepv.o:	HGRD_DEFN.o RXNS_DATA_MODULE.o CGRID_SPCS.o UTILIO_DEFN.o \
		$(BASE_INC)/FILES_CTM.EXT
SEDIMENTATION.o:	CGRID_SPCS.o GRID_CONF.o ASX_DATA_MOD.o VDIFF_MAP.o \
		UTILIO_DEFN.o VDIFF_DIAG.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
tri.o:	VGRD_DEFN.o CGRID_SPCS.o
vdiffacmx.o:	CGRID_SPCS.o GRID_CONF.o EMIS_DEFN.o DEPV_DEFN.o \
		ASX_DATA_MOD.o VDIFF_MAP.o UTILIO_DEFN.o BIDI_MOD.o LSM_MOD.o \
		VDIFF_DIAG.o $(BASE_INC)/FILES_CTM.EXT
VDIFF_DIAG.o:	UTILIO_DEFN.o GRID_CONF.o CGRID_SPCS.o se_modules.o \
		$(BASE_INC)/FILES_CTM.EXT
VDIFF_MAP.o:	CGRID_SPCS.o AERO_EMIS.o UTILIO_DEFN.o $(BASE_INC)/EMISPRM.EXT
vdiffproc.o:	CGRID_SPCS.o GRID_CONF.o EMIS_DEFN.o DEPV_DEFN.o \
		ASX_DATA_MOD.o VDIFF_MAP.o UTILIO_DEFN.o BIDI_MOD.o HGSIM.o \
		LSM_MOD.o SEDIMENTATION.o VDIFF_DIAG.o PA_DEFN.o $(BASE_INC)/FILES_CTM.EXT
ABFLUX_MOD.o:	HGRD_DEFN.o UTILIO_DEFN.o BIDI_MOD.o ASX_DATA_MOD.o \
		MOSAIC_MOD.o LSM_MOD.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
BIDI_MOD.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
cgrid_depv.o:	CGRID_SPCS.o HGRD_DEFN.o UTILIO_DEFN.o
DEPV_DEFN.o:	HGRD_DEFN.o RXNS_DATA_MODULE.o CGRID_SPCS.o DEPVVARS.o \
		UTILIO_DEFN.o MOSAIC_MOD.o LSM_MOD.o BIDI_MOD.o ASX_DATA_MOD.o \
		$(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
gas_depv_map.o:	CGRID_SPCS.o DEPVVARS.o UTILIO_DEFN.o
HGSIM.o:	HGRD_DEFN.o UTILIO_DEFN.o ASX_DATA_MOD.o LSM_MOD.o BIDI_MOD.o \
		DEPVVARS.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
LSM_MOD.o:	HGRD_DEFN.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
m3dry.o:	HGRD_DEFN.o DEPVVARS.o VGRD_DEFN.o UTILIO_DEFN.o se_modules.o \
		ABFLUX_MOD.o LSM_MOD.o MOSAIC_MOD.o ASX_DATA_MOD.o BIDI_MOD.o \
		HGSIM.o $(BASE_INC)/PE_COMM.EXT $(BASE_INC)/FILES_CTM.EXT
MOSAIC_MOD.o:	HGRD_DEFN.o LSM_MOD.o UTILIO_DEFN.o ASX_DATA_MOD.o \
		$(BASE_INC)/FILES_CTM.EXT
opdepv_diag.o:	GRID_CONF.o DEPVVARS.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
opdepv_fst.o:	GRID_CONF.o DEPVVARS.o UTILIO_DEFN.o LSM_MOD.o $(BASE_INC)/FILES_CTM.EXT
opdepv_mos.o:	GRID_CONF.o DEPVVARS.o UTILIO_DEFN.o LSM_MOD.o $(BASE_INC)/FILES_CTM.EXT
BEIS_DEFN.o:	HGRD_DEFN.o CGRID_SPCS.o BIOG_EMIS.o UTILIO_DEFN.o \
		$(BASE_INC)/CONST.EXT
BIOG_EMIS.o:	UTILIO_DEFN.o
cropcal.o:	HGRD_DEFN.o UTILIO_DEFN.o
DUST_EMIS.o:	LUS_DEFN.o AERO_DATA.o HGRD_DEFN.o ASX_DATA_MOD.o \
		UTILIO_DEFN.o GRID_CONF.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
EMIS_DEFN.o:	GRID_CONF.o CGRID_SPCS.o BEIS_DEFN.o MGEMIS.o LTNG_DEFN.o \
		PT3D_DEFN.o UTILIO_DEFN.o AERO_EMIS.o AERO_DATA.o DEPV_DEFN.o \
		BIOG_EMIS.o PTMAP.o ASX_DATA_MOD.o RXNS_DATA_MODULE.o STK_EMIS.o \
		$(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
LTNG_DEFN.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/CONST.EXT \
		$(BASE_INC)/FILES_CTM.EXT
LUS_DEFN.o:	HGRD_DEFN.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
MGEMIS.o:	HGRD_DEFN.o UTILIO_DEFN.o CGRID_SPCS.o RXNS_DATA_MODULE.o \
		GRID_CONF.o AEROMET_DATA.o ASX_DATA_MOD.o PCGRID_DEFN.o $(BASE_INC)/CONST.EXT \
		$(BASE_INC)/FILES_CTM.EXT
opemis.o:	VGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o RXNS_DATA_MODULE.o \
		EMIS_DEFN.o $(BASE_INC)/FILES_CTM.EXT
PT3D_DEFN.o:	UDTYPES.o GRID_CONF.o CGRID_SPCS.o STK_PRMS.o STK_EMIS.o \
		PTMET.o PTBILIN.o PTMAP.o UTILIO_DEFN.o bmatvec.o RXNS_DATA_MODULE.o \
		VGRD_DEFN.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
PTBILIN.o:	UDTYPES.o HGRD_DEFN.o VGRD_DEFN.o STK_PRMS.o
PTMAP.o:	UDTYPES.o CGRID_SPCS.o STK_EMIS.o AERO_DATA.o UTILIO_DEFN.o
PTMET.o:	UDTYPES.o STK_PRMS.o PTBILIN.o bmatvec.o UTILIO_DEFN.o \
		VGRD_DEFN.o $(BASE_INC)/FILES_CTM.EXT
SSEMIS.o:	HGRD_DEFN.o AERO_DATA.o UTILIO_DEFN.o se_modules.o GRID_CONF.o \
		AEROMET_DATA.o ASX_DATA_MOD.o $(BASE_INC)/FILES_CTM.EXT
STK_EMIS.o:	UDTYPES.o UTILIO_DEFN.o
STK_PRMS.o:	UDTYPES.o HGRD_DEFN.o UTILIO_DEFN.o
tfabove.o:	HGRD_DEFN.o ASX_DATA_MOD.o LUS_DEFN.o
tfbelow.o:	HGRD_DEFN.o LUS_DEFN.o UTILIO_DEFN.o
beis3.o:	BIOG_EMIS.o ASX_DATA_MOD.o
chkgrid.o:	UTILIO_DEFN.o
czangle.o:	UTILIO_DEFN.o ASX_DATA_MOD.o $(BASE_INC)/CONST.EXT
hrno.o:	HGRD_DEFN.o BIOG_EMIS.o ASX_DATA_MOD.o LSM_MOD.o UTILIO_DEFN.o \
		se_modules.o
parsline.o:	UTILIO_DEFN.o
tmpbeis.o:	GRID_CONF.o UTILIO_DEFN.o BIOG_EMIS.o se_modules.o
wrdaymsg.o:	UTILIO_DEFN.o
fire_plmris.o:	$(BASE_INC)/CONST.EXT
openlayout.o:	HGRD_DEFN.o UTILIO_DEFN.o
oppt3d_diag.o:	GRID_CONF.o UTILIO_DEFN.o
plmris.o:	$(BASE_INC)/CONST.EXT
preplm.o:	UTILIO_DEFN.o
ungridb2.o:	HGRD_DEFN.o PTBILIN.o
CGRID_SPCS.o:	UTILIO_DEFN.o
AERO_PHOTDATA.o:	GRID_CONF.o UTILIO_DEFN.o AERO_DATA.o SOA_DEFN.o \
		AEROMET_DATA.o CSQY_DATA.o RXNS_DATA_MODULE.o CGRID_SPCS.o twoway_rrtmg_aero_optics.o \
		$(BASE_INC)/CONST.EXT
CLOUD_OPTICS.o:	GRID_CONF.o UTILIO_DEFN.o CSQY_DATA.o
CSQY_DATA.o:	UTILIO_DEFN.o RXNS_DATA_MODULE.o AERO_DATA.o GRID_CONF.o
o3totcol.o:	UTILIO_DEFN.o
opphot.o:	GRID_CONF.o RXNS_DATA_MODULE.o UTILIO_DEFN.o PHOT_MET_DATA.o \
		PHOT_MOD.o $(BASE_INC)/FILES_CTM.EXT
phot.o:	RXNS_DATA_MODULE.o CGRID_SPCS.o PCGRID_DEFN.o UTILIO_DEFN.o \
		AERO_DATA.o PHOT_MOD.o AERO_PHOTDATA.o PHOTOLYSIS_ALBEDO.o PHOT_MET_DATA.o \
		CLOUD_OPTICS.o SEAS_STRAT_O3_MIN.o se_modules.o $(BASE_INC)/FILES_CTM.EXT
PHOT_MET_DATA.o:	GRID_CONF.o UTILIO_DEFN.o CLOUD_OPTICS.o $(BASE_INC)/CONST.EXT \
		$(BASE_INC)/FILES_CTM.EXT
PHOT_MOD.o:	CSQY_DATA.o UTILIO_DEFN.o RXNS_DATA_MODULE.o CLOUD_OPTICS.o \
		AERO_PHOTDATA.o $(BASE_INC)/CONST.EXT
PHOTOLYSIS_ALBEDO.o:	UTILIO_DEFN.o se_modules.o PHOT_MOD.o PCGRID_DEFN.o \
		LSM_MOD.o $(BASE_INC)/FILES_CTM.EXT
SEAS_STRAT_O3_MIN.o:	GRID_CONF.o UTILIO_DEFN.o
twoway_rrtmg_aero_optics.o:	complex_number_module.o
hrdriver.o:	HGRD_DEFN.o VGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o \
		RXNS_DATA_MODULE.o AEROSOL_CHEMISTRY.o RXNS_FUNC_MODULE.o hrdata_mod.o \
		PA_DEFN.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT $(BASE_INC)/EMISPRM.EXT
chem_py.o:  forpy_mod.o HGRD_DEFN.o VGRD_DEFN.o CGRID_SPCS.o UTILIO_DEFN.o \
		RXNS_DATA_MODULE.o AEROSOL_CHEMISTRY.o RXNS_FUNC_MODULE.o hrdata_mod.o \
		PA_DEFN.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT $(BASE_INC)/EMISPRM.EXT       
resnet.o:    forpy_mod.o UTILIO_DEFN.o
hrg1.o:	hrdata_mod.o
hrg2.o:	hrdata_mod.o
hrg3.o:	hrdata_mod.o
hrg4.o:	hrdata_mod.o
hrinit.o:	UTILIO_DEFN.o hrdata_mod.o RXNS_DATA_MODULE.o RXNS_FUNC_MODULE.o \
		$(BASE_INC)/CONST.EXT
hrprodloss.o:	hrdata_mod.o
hrrates.o:	hrdata_mod.o
hrsolver.o:	UTILIO_DEFN.o RXNS_DATA_MODULE.o hrdata_mod.o
RXNS_FUNC_MODULE.o:	RXNS_DATA_MODULE.o UTILIO_DEFN.o CGRID_SPCS.o
AERO_DATA.o:	RXNS_DATA_MODULE.o CGRID_SPCS.o AEROMET_DATA.o UTILIO_DEFN.o \
		$(BASE_INC)/FILES_CTM.EXT
aero_depv.o:	GRID_CONF.o RXNS_DATA_MODULE.o CGRID_SPCS.o UTILIO_DEFN.o \
		AERO_DATA.o AEROMET_DATA.o MOSAIC_MOD.o LSM_MOD.o ASX_DATA_MOD.o \
		$(BASE_INC)/FILES_CTM.EXT
aero_driver.o:	GRID_CONF.o RXNS_DATA_MODULE.o AERO_DATA.o PRECURSOR_DATA.o \
		SOA_DEFN.o AEROMET_DATA.o AOD_DEFN.o UTILIO_DEFN.o CGRID_SPCS.o \
		AEROSOL_CHEMISTRY.o PMDIAG_DATA.o $(BASE_INC)/FILES_CTM.EXT
AERO_EMIS.o:	AERO_DATA.o GRID_CONF.o RXNS_DATA_MODULE.o CGRID_SPCS.o \
		SSEMIS.o DUST_EMIS.o PRECURSOR_DATA.o UTILIO_DEFN.o PT3D_DEFN.o \
		PTMAP.o ASX_DATA_MOD.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
AEROMET_DATA.o:	$(BASE_INC)/CONST.EXT
AEROSOL_CHEMISTRY.o:	UTILIO_DEFN.o GRID_CONF.o RXNS_DATA_MODULE.o \
		AERO_DATA.o AEROMET_DATA.o PRECURSOR_DATA.o SOA_DEFN.o $(BASE_INC)/CONST.EXT
aero_subs.o:	AERO_DATA.o PRECURSOR_DATA.o SOA_DEFN.o AEROMET_DATA.o \
		UTILIO_DEFN.o AOD_DEFN.o $(BASE_INC)/CONST.EXT
AOD_DEFN.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o AERO_DATA.o \
		ASX_DATA_MOD.o $(BASE_INC)/FILES_CTM.EXT
getpar.o:	AERO_DATA.o AEROMET_DATA.o
opapmdiag.o:	GRID_CONF.o UTILIO_DEFN.o PMDIAG_DATA.o $(BASE_INC)/FILES_CTM.EXT
opavis.o:	GRID_CONF.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
oppmdiag.o:	GRID_CONF.o UTILIO_DEFN.o RXNS_DATA_MODULE.o PMDIAG_DATA.o \
		$(BASE_INC)/FILES_CTM.EXT
opvis.o:	GRID_CONF.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
PRECURSOR_DATA.o:	RXNS_DATA_MODULE.o AERO_DATA.o CGRID_SPCS.o \
		UTILIO_DEFN.o AEROMET_DATA.o
SOA_DEFN.o:	RXNS_DATA_MODULE.o AERO_DATA.o AEROMET_DATA.o CGRID_SPCS.o \
		UTILIO_DEFN.o
aqchem.o:	RXNS_DATA_MODULE.o AQ_DATA.o AERO_DATA.o UTILIO_DEFN.o \
		$(BASE_INC)/CONST.EXT
AQ_DATA.o:	AERO_DATA.o CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/CONST.EXT
aq_map.o:	CGRID_SPCS.o AERO_DATA.o AQ_DATA.o UTILIO_DEFN.o $(BASE_INC)/CONST.EXT
cldproc_acm.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/CONST.EXT \
		$(BASE_INC)/FILES_CTM.EXT
convcld_acm.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o AQ_DATA.o \
		PHOT_MET_DATA.o ASX_DATA_MOD.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT
getalpha.o:	AERO_DATA.o $(BASE_INC)/CONST.EXT
hlconst.o:	UTILIO_DEFN.o
opwdep.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/CONST.EXT \
		$(BASE_INC)/FILES_CTM.EXT
rescld.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/CONST.EXT \
		$(BASE_INC)/FILES_CTM.EXT
scavwdep.o:	CGRID_SPCS.o UTILIO_DEFN.o $(BASE_INC)/CONST.EXT
pa_compmech.o:	PA_GLOBAL.o PA_VARS.o PA_PARSE.o
pa_datagen.o:	UTILIO_DEFN.o PA_VARS.o PA_DEFN.o
pa_errcheck.o:	PA_GLOBAL.o CGRID_SPCS.o PA_IPRVARS.o PA_VARS.o \
		PA_PARSE.o PA_IPRDEF.o PA_DEFN.o
pa_getcoef.o:	PA_GLOBAL.o PA_PARSE.o
pa_getcycle.o:	PA_VARS.o PA_PARSE.o
pa_getdesc.o:	PA_GLOBAL.o PA_VARS.o PA_DEFN.o PA_PARSE.o
pa_getfamily.o:	PA_VARS.o PA_PARSE.o
pa_getiprout.o:	PA_DEFN.o PA_IPRVARS.o PA_PARSE.o
pa_getirrout.o:	PA_GLOBAL.o PA_VARS.o PA_DEFN.o PA_PARSE.o
pa_getrxns.o:	PA_GLOBAL.o PA_VARS.o PA_PARSE.o
pa_getrxnsum.o:	PA_GLOBAL.o PA_VARS.o PA_PARSE.o
PA_GLOBAL.o:	RXNS_DATA_MODULE.o
pa_init.o:	GRID_CONF.o PAGRD_DEFN.o UTILIO_DEFN.o se_modules.o \
		PA_DEFN.o $(BASE_INC)/FILES_CTM.EXT
PA_IPRDEF.o:	PA_DEFN.o
PA_IPRVARS.o:	PA_GLOBAL.o PA_VARS.o
pa_irr_ctl.o:	GRID_CONF.o CGRID_SPCS.o PA_DEFN.o PAGRD_DEFN.o
pa_irr.o:	GRID_CONF.o RXNS_DATA_MODULE.o CGRID_SPCS.o PA_DEFN.o \
		UTILIO_DEFN.o
pa_mkhdr.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o PA_DEFN.o PAGRD_DEFN.o
pa_molcloss.o:	PA_GLOBAL.o PA_VARS.o
pa_molcprod.o:	PA_GLOBAL.o PA_VARS.o
pa_output.o:	GRID_CONF.o PAGRD_DEFN.o UTILIO_DEFN.o se_modules.o \
		PA_DEFN.o $(BASE_INC)/FILES_CTM.EXT
pa_read.o:	PA_GLOBAL.o PA_VARS.o PA_DEFN.o PA_PARSE.o
pa_report.o:	PA_GLOBAL.o PA_VARS.o PA_IPRVARS.o PA_DEFN.o PA_PARSE.o
pa_setup_ipr.o:	PA_GLOBAL.o PA_VARS.o PA_IPRVARS.o PA_IPRDEF.o \
		PA_DEFN.o PA_PARSE.o
pa_setup_irr.o:	PA_GLOBAL.o PA_DEFN.o PA_VARS.o PA_PARSE.o
pa_update.o:	GRID_CONF.o CGRID_SPCS.o PA_DEFN.o PAGRD_DEFN.o UTILIO_DEFN.o \
		se_modules.o $(BASE_INC)/CONST.EXT $(BASE_INC)/FILES_CTM.EXT \
		$(BASE_INC)/EMISPRM.EXT
PA_VARS.o:	PA_GLOBAL.o CGRID_SPCS.o
pa_wrtpadefn.o:	PA_VARS.o PA_IPRVARS.o PA_PARSE.o PA_DEFN.o
cksummer.o:	GRID_CONF.o CGRID_SPCS.o UTILIO_DEFN.o se_modules.o
lstepf.o:	UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
setup_logdev.o:	UTILIO_DEFN.o
subhdomain.o:	UTILIO_DEFN.o
subhfile.o:	GRID_CONF.o UTILIO_DEFN.o $(BASE_INC)/FILES_CTM.EXT
UTILIO_DEFN.o:	parutilio.o
