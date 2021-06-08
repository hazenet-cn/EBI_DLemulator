      SUBROUTINE CHEM_PY (CGRID, JDATE, JTIME, TSTEP )

      USE HGRD_DEFN             ! horizontal domain specifications
      USE VGRD_DEFN             ! vertical layer specifications
      USE CGRID_SPCS            ! CGRID mechanism species
      USE UTILIO_DEFN           ! IOAPI parameters and functions declarations
      USE RXNS_DATA  
      USE AEROSOL_CHEMISTRY
      USE RXNS_FUNCTION
      USE HRDATA
      USE PA_DEFN, Only: LIRR   ! Process Anaylsis control and data variable
      use forpy_mod
      
      IMPLICIT NONE

!..Includes:
      INCLUDE SUBST_FILES_ID  ! CMAQ files
      INCLUDE SUBST_CONST     ! CMAQ constants

      INCLUDE SUBST_EMISPRM   ! Emissions processing control parameters

!..Arguments:
      REAL, POINTER :: CGRID( :,:,:,: )  ! Species concentrations
      INTEGER JDATE           ! Current date (YYYYDDD)
      INTEGER JTIME           ! Current time (HHMMSS)
      INTEGER TSTEP( 3 )      ! Time step vector (HHMMSS)
      integer :: time1, time2      

!..for forpy:
      integer :: ierror
      type(module_py) :: np
      type(tuple) :: args
      type(object) :: ymax
      type(ndarray) :: ymax_nd
      real, dimension(:,:,:), pointer ::ymaxf
      
!..Saved Local Variables:

      CHARACTER( 16 ), SAVE :: PNAME = 'CHEM_PY'     ! Program name

!..Scratch Local Variables:
      CHARACTER( 132 ) :: XMSG           ! Message text
      CHARACTER(  16 ) :: VNAME         ! Name of I/O API data variable
    
      INTEGER       :: C, R, L, S, V, N, T   ! loop induction variables
      INTEGER       :: DATA_NUM, VAR_NUM 
      INTEGER, PARAMETER :: MET_NUM = 43   ! number of meteorological parameter            
      
      INTEGER ITMSTEP         ! Chemistry integration interval (sec)
      INTEGER MIDDATE         ! Date at time step midpoint
      INTEGER MIDTIME         ! Time at time step midpoint
      INTEGER NPH             ! Index for number of phot. rxns in PHOT
      INTEGER SPC             ! Species loop index
      
      LOGICAL, SAVE :: LFIRST = .TRUE.  ! Flag for first call to this subroutine
      INTEGER       :: ALLOCSTAT

      INTEGER       :: GXOFF, GYOFF            ! global origin offset from file
      INTEGER, SAVE :: PECOL_OFFSET        ! Local Column Offset for processor
      INTEGER, SAVE :: PEROW_OFFSET        ! Local Column Offset for processor
      
! for INTERPX
      INTEGER, SAVE :: STRTCOLMC2, ENDCOLMC2, STRTROWMC2, ENDROWMC2
      INTEGER, SAVE :: STRTCOLMC3, ENDCOLMC3, STRTROWMC3, ENDROWMC3

      INTERFACE
         SUBROUTINE PHOT ( MDATE, MTIME, JDATE, JTIME, DTSTEP, RJ, COSUN)
          INTEGER, INTENT( IN )  :: MDATE       !"centered" Julian date (YYYYDDD)
          INTEGER, INTENT( IN )  :: MTIME       !"centered" time (HHMMSS)
          INTEGER, INTENT( IN )  :: JDATE       ! Current date (YYYYDDD)
          INTEGER, INTENT( IN )  :: JTIME       ! Current time (HHMMSS)
          INTEGER, INTENT( IN )  :: DTSTEP( : ) ! time step vector (HHMMSS)
          REAL,    INTENT( OUT ) :: RJ(:,:,:,:) ! gridded J-values  (1/min units)
          REAL,    INTENT( OUT ) :: COSUN(:,:) 
        END SUBROUTINE PHOT
         SUBROUTINE resnet(PYTHON_IN, PYTHON_OUT)
          REAL, INTENT( IN )  :: PYTHON_IN(:,:)
		  REAL, INTENT( OUT ) :: PYTHON_OUT(:,:)
          END SUBROUTINE resnet
      END INTERFACE

!**********************************************************************

      IF( NUMB_MECH_SPC .EQ. 0 ) RETURN

      IF( LFIRST ) THEN

         LOGDEV = INIT3( )

         IF( MECHNAME .NE. 'SAPRC07TC_AE6_AQ' ) THEN
             XMSG = 'This version of the EBI solver can only be used with'
     &            // ' the SAPRC07TC_AE6_AQ chemical mechanism'
             CALL M3EXIT( PNAME, 0, 0, XMSG, XSTAT1 )
         END IF
         
         !DO SPC = 1, NUMB_MECH_SPC
            !WRITE( LOGDEV, * ) "First in chem_py", SPC, CHEMISTRY_SPC( SPC )
         !END DO                      
         
         ! get python array        
         DATA_NUM = MY_NCOLS * MY_NROWS * NLAYS
         VAR_NUM  = NUMB_MECH_SPC + MET_NUM 
         ALLOCATE( D_MAX_IN(DATA_NUM, NUMB_MECH_SPC),  D_MIN_IN(DATA_NUM, NUMB_MECH_SPC),
     &             PYTHON_IN(DATA_NUM, VAR_NUM), PYTHON_OUT(DATA_NUM, NUMB_MECH_SPC),  
     &             STAT = ALLOCSTAT)
         IF ( ALLOCSTAT .NE. 0 ) THEN
             XMSG = 'Failure allocating PYTHON'
             CALL M3EXIT( PNAME, JDATE, JTIME, XMSG, XSTAT1 )
         END IF
         
         ! get met_data array                    
         CALL SUBHFILE ( MET_CRO_3D, GXOFF, GYOFF, STRTCOLMC3, ENDCOLMC3, STRTROWMC3, ENDROWMC3 )

         ALLOCATE( DENS( NCOLS, NROWS, NLAYS ), PRES( NCOLS, NROWS, NLAYS ),
     &             QV  ( NCOLS, NROWS, NLAYS ), TA  ( NCOLS, NROWS, NLAYS ),
     &             RJ  ( NCOLS, NROWS, NLAYS, NPHOTAB ), COSUN( NCOLS, NROWS ),
     &             y_max (NUMB_MECH_SPC, NCOLS, NROWS)))  
     
         LFIRST = .FALSE.

      END IF      ! First time

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!  Set date and time to center of time step, get necessary physical
!  data, and get photolysis rates
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      
      XMSG = 'Begin: chem_py'
	  WRITE( LOGDEV, *) XMSG
      
!.. Get meterological variables     
      MIDDATE = JDATE
      MIDTIME = JTIME
      ITMSTEP = TIME2SEC( TSTEP( 2 ) )
      CALL NEXTIME( MIDDATE, MIDTIME, SEC2TIME( ITMSTEP / 2 ) )           

      VNAME = 'TA'
      IF ( .NOT. INTERPX( MET_CRO_3D, VNAME, PNAME,
     &                    STRTCOLMC3, ENDCOLMC3, STRTROWMC3, ENDROWMC3, 1,NLAYS,
     &                    MIDDATE, MIDTIME, TA ) ) THEN
         XMSG = 'Could not read TA from MET_CRO_3D'
         CALL M3EXIT( PNAME, JDATE, JTIME, XMSG, XSTAT1 )
      END IF

      VNAME = 'QV'
      IF ( .NOT. INTERPX( MET_CRO_3D, VNAME, PNAME,
     &                    STRTCOLMC3, ENDCOLMC3, STRTROWMC3, ENDROWMC3, 1,NLAYS,
     &                    MIDDATE, MIDTIME, QV ) ) THEN
         XMSG = 'Could not read QV from MET_CRO_3D'
         CALL M3EXIT( PNAME, JDATE, JTIME, XMSG, XSTAT1 )
      END IF

       VNAME = 'DENS'
       IF ( .NOT. INTERPX( MET_CRO_3D, VNAME, PNAME,
     &                    STRTCOLMC3, ENDCOLMC3, STRTROWMC3, ENDROWMC3, 1,NLAYS,
     &                    MIDDATE, MIDTIME, DENS ) ) THEN
          XMSG = 'Could not read DENS from MET_CRO_3D'
          CALL M3EXIT( PNAME, JDATE, JTIME, XMSG, XSTAT1 )
       END IF

      VNAME = 'PRES'
      IF ( .NOT. INTERPX( MET_CRO_3D, VNAME, PNAME,
     &                    STRTCOLMC3, ENDCOLMC3, STRTROWMC3, ENDROWMC3, 1,NLAYS,
     &                    MIDDATE, MIDTIME, PRES ) ) THEN
         XMSG = 'Could not read PRES from MET_CRO_3D'
         CALL M3EXIT ( PNAME, JDATE, JTIME, XMSG, XSTAT1 )
      END IF
      
      XMSG = 'End: import meterological variables'
	  WRITE( LOGDEV, *) XMSG

!.. Get photolysis rates in 1/min
      CALL PHOT ( MIDDATE, MIDTIME, JDATE, JTIME, TSTEP, RJ, COSUN )
!.. Get Heteorogeneous rates and Update Aerosol Distribution Properties
      CALL HETCHEM_RATES( TA, PRES, QV, CGRID )     

!.. Python data: the maximum value of historical data for each grid cell in the simulation area
!.. If you cannot get the data, you can delete the corresponding code and use only the maximum value of the training data   
      ierror = import_py(np, "numpy")     
      ierror = tuple_create(args, 1)
      ierror = args%setitem(0, "y_max.npy")
      ierror = call_py(ymax, np, "load", args)
      ierror = cast(ymax_nd, ymax)
      ierror = ymax_nd%get_data(ymaxf, order='C')     	
      y_max = ymaxf(:, STRTCOLMC3:ENDCOLMC3, STRTROWMC3:ENDROWMC3)
      
      XMSG = 'END: python data'
	  WRITE( LOGDEV, *) XMSG
      
      N = 0
      T = NUMB_MECH_SPC + 1
      DO L = 1, NLAYS
         DO R = 1, MY_NROWS
            DO C = 1, MY_NCOLS
               N = N + 1
               DO V = 1, NUMB_MECH_SPC
                  S = CGRID_INDEX( V )				  
                  PYTHON_IN(N,V) = CGRID(C,R,L,S)                 
				  IF ( ISNAN(PYTHON_IN(N,V))) THEN
				     WRITE( LOGDEV, *) N, V
                     XMSG = 'NAN ERROR HAPPEN, EXIT!'
                     CALL M3EXIT ( PNAME, JDATE, JTIME, XMSG, XSTAT1 )
				  END IF
               END DO 
               PYTHON_IN(N,T)      = TA(C,R,L)
               PYTHON_IN(N,T+1)    = QV(C,R,L)
               PYTHON_IN(N,T+2)    = PRES(C,R,L)
               PYTHON_IN(N,T+3)    = DENS(C,R,L)
               PYTHON_IN(N,T+4)    = COSUN(C,R)
               FORALL( NPH = 1:NPHOTAB ) PYTHON_IN(N,T+4+NPH) = RJ( C,R,L,NPH )               
            END DO
         END DO
      END DO
	  
      XMSG = 'Begin: dl'
	  WRITE( LOGDEV, *) XMSG

	  call system_clock(time1)
      CALL resnet(PYTHON_IN, PYTHON_OUT)    
	  call system_clock(time2)
      write(logdev,*) "dl time: ", (real(time2)-real(time1))/10000
      
      XMSG = 'End: dl'
	  WRITE( LOGDEV, *) XMSG
	  
      N = 0
      DO L = 1, NLAYS
         DO R = 1, MY_NROWS
             DO C = 1, MY_NCOLS
                N = N + 1
                DO V = 1, NUMB_MECH_SPC
                   S = CGRID_INDEX( V )                   				   
                   CGRID(C,R,L,S) = PYTHON_OUT(N,V)                                     
                   IF (CGRID(C,R,L,S) > y_max(V,C,R)) THEN
                       CGRID(C,R,L,S) = y_max(V,C,R)
                   END IF                   
                END DO
             END DO 
         END DO 
      END DO
	  
      XMSG = 'End: chem_py'
	  WRITE( LOGDEV, *) XMSG
      
      ! Update Aerosol Surface Area
      CALL HETCHEM_UPDATE_AERO( CGRID )
          
      RETURN
      END SUBROUTINE
  
      
      
      
