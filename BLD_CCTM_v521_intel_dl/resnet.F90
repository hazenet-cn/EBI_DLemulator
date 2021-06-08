      subroutine resnet(python_in, python_out)
      
      ! use StandardScaler in data pre-processing
      ! use keras
      ! model of difference resnet
      use forpy_mod
      use UTILIO_DEFN
     
      implicit none

      integer :: ierror
      
      INTEGER, SAVE :: LOGDEV       ! FORTRAN unit number for log file
      INTEGER       :: ALLOCSTAT
      LOGICAL, SAVE :: FIRSTIME = .TRUE.
      CHARACTER( 96 ) :: XMSG = ' '
      integer :: time1, time2
      
      integer :: xcol, xrow, i, j
      type(module_py) :: np, k
      type(tuple) :: args 
      type(dict) :: kwargs      
      type(object) :: X_train_min, X_train_max, Y_train_min, Y_train_max, Y_train_min_d, Y_train_max_d
      type(object) :: model, predict_results_std
      type(ndarray) :: X_train_min_nd, X_train_max_nd, Y_train_min_nd, Y_train_max_nd, Y_train_min_d_nd, Y_train_max_d_nd
      type(ndarray) :: python_nd, predict_results_std_nd
      real, dimension(:), pointer :: X_train_minf, X_train_maxf, Y_train_minf, Y_train_maxf, Y_train_min_df, Y_train_max_df
      real, dimension(:,:), pointer :: predict_results_stdf 
      real, dimension(:,:), allocatable :: temp, python_std, t
      
      real, intent(in)  :: python_in(:,:)
      real, intent(out) :: python_out(:,:)

      nullify(X_train_maxf)
      nullify(X_train_minf)
      nullify(Y_train_minf)
      nullify(Y_train_maxf)
      nullify(Y_train_min_df)
      nullify(Y_train_max_df)
      nullify(predict_results_stdf)
      
      IF ( FIRSTIME ) THEN
         LOGDEV = INIT3 ()
         FIRSTIME = .FALSE.
         
         XMSG = 'Firstime in resnet'
         WRITE( LOGDEV, *) XMSG
      END IF

      ierror = import_py(np, "numpy")
      
      ierror = tuple_create(args, 1)
	  ierror = args%setitem(0, "/lustre/home/acct-esehazenet/hazenet-pg3/CMAQv5.2.1AIr/data/python/dl/X_train_min.npy")
	  ierror = call_py(X_train_min, np, "load", args)
	  ierror = cast(X_train_min_nd, X_train_min)
	  ierror = X_train_min_nd%get_data(X_train_minf)
	 
	  ierror = tuple_create(args, 1)
	  ierror = args%setitem(0, "/lustre/home/acct-esehazenet/hazenet-pg3/CMAQv5.2.1AIr/data/python/dl/X_train_max.npy")
	  ierror = call_py(X_train_max, np, "load", args)
	  ierror = cast(X_train_max_nd, X_train_max)
	  ierror = X_train_max_nd%get_data(X_train_maxf)
      
      ierror = tuple_create(args, 1)
      ierror = args%setitem(0, "/lustre/home/acct-esehazenet/hazenet-pg3/CMAQv5.2.1AIr/data/python/dl/Y_train_min.npy")
      ierror = call_py(Y_train_min, np, "load", args)
      ierror = cast(Y_train_min_nd, Y_train_min)
      ierror = Y_train_min_nd%get_data(Y_train_minf)

      ierror = tuple_create(args, 1)
      ierror = args%setitem(0, "/lustre/home/acct-esehazenet/hazenet-pg3/CMAQv5.2.1AIr/data/python/dl/Y_train_max.npy")
      ierror = call_py(Y_train_max, np, "load", args)
      ierror = cast(Y_train_max_nd, Y_train_max)
      ierror = Y_train_max_nd%get_data(Y_train_maxf)
      
      ierror = tuple_create(args, 1)
      ierror = args%setitem(0, "/lustre/home/acct-esehazenet/hazenet-pg3/CMAQv5.2.1AIr/data/python/dl/Y_train_min_d.npy")
      ierror = call_py(Y_train_min_d, np, "load", args)
      ierror = cast(Y_train_min_d_nd, Y_train_min_d)
      ierror = Y_train_min_d_nd%get_data(Y_train_min_df)

      ierror = tuple_create(args, 1)
      ierror = args%setitem(0, "/lustre/home/acct-esehazenet/hazenet-pg3/CMAQv5.2.1AIr/data/python/dl/Y_train_max_d.npy")
      ierror = call_py(Y_train_max_d, np, "load", args)
      ierror = cast(Y_train_max_d_nd, Y_train_max_d)
      ierror = Y_train_max_d_nd%get_data(Y_train_max_df)
      
      XMSG = 'Loaded numpy data'
      WRITE( LOGDEV, *) XMSG
         
      xrow = size(python_in, 1) 
      xcol = size(python_in, 2)
      WRITE( LOGDEV, *) xrow, xcol
     
      allocate(python_std(xrow,xcol), STAT = ALLOCSTAT)
      
      IF ( ALLOCSTAT .NE. 0 ) THEN
          XMSG = 'Failure allocating python_std'
          WRITE( LOGDEV, *) XMSG
      END IF
      XMSG = 'at 87'
      WRITE( LOGDEV, *) XMSG

      do j = 1, xcol
          do i = 1, xrow
              if ( (X_train_maxf(j)- X_train_minf(j)) .ne. 0) then
		          python_std(i,j) = (python_in(i,j) - X_train_minf(j)) / (X_train_maxf(j)- X_train_minf(j))
              else
                  python_std(i,j) = X_train_minf(j)
              end if
          end do     
	  end do
      
      ierror = ndarray_create(python_nd, python_std)
      XMSG = 'at 108'
      WRITE( LOGDEV, *) XMSG
      !WRITE( LOGDEV, *) python_std(1,:)
      
      ierror = import_py(k, "keras.models")
      ierror = tuple_create(args, 1)
      ierror = args%setitem(0, "/lustre/home/acct-esehazenet/hazenet-pg3/CMAQv5.2.1AIr/data/python/dl/resnet.h5")
      ierror = call_py(model, k, "load_model", args)
     
      XMSG = 'end load model'
      WRITE( LOGDEV, *) XMSG
      
      
      ierror = tuple_create(args, 1)
      ierror = args%setitem(0, python_nd)
      ierror = dict_create(kwargs)
      ierror = kwargs%setitem("batch_size", 4096)
      
      call system_clock(time1)      
      ierror = call_py(predict_results_std, model, "predict", args, kwargs)    
      call system_clock(time2)
      write(logdev,*) "predict time: ", (real(time2)-real(time1))/10000
      ierror = cast(predict_results_std_nd, predict_results_std)
      ierror = predict_results_std_nd%get_data(predict_results_stdf, order='C')     
          
      XMSG = 'get predict_result'
      WRITE( LOGDEV, *) XMSG
          
      xrow = size(predict_results_stdf, 2)
      xcol= size(predict_results_stdf, 1)
      WRITE( LOGDEV, *) xrow, xcol
      
      allocate(temp(xrow,xcol), t(xrow,xcol), STAT = ALLOCSTAT)      
      temp = transpose(predict_results_stdf)
            
      do j = 1, xcol
          do i = 1, xrow
             t(i,j) = temp(i,j) * (Y_train_max_df(j) - Y_train_min_df(j)) + Y_train_min_df(j)
             if (t(i,j) < Y_train_min_df(j)) then
                 t(i,j) = Y_train_min_df(j)
             elseif (t(i,j) > Y_train_max_df(j)) then
                 t(i,j) = Y_train_max_df(j)
             end if	
             python_out(i,j) = t(i,j) + python_in(i,j)  
             if ( python_out(i,j) < Y_train_minf(j) ) then
                 python_out(i,j) = Y_train_minf(j)                
!             elseif (t(i,j) > Y_train_maxf(j)) then
!                 t(i,j) = Y_train_maxf(j)
             end if	
         end do             
      end do

      nullify(X_train_maxf)
      nullify(X_train_minf)
      nullify(Y_train_min_df)
      nullify(Y_train_max_df)
      nullify(predict_results_stdf)
      deallocate(temp)   
      deallocate(t)      
      deallocate(python_std)
     
      call np%destroy
      call k%destroy
      call args%destroy
      call kwargs%destroy
      call model%destroy
      call X_train_min%destroy
      call X_train_max%destroy
      call Y_train_min_d%destroy
	  call Y_train_max_d%destroy      
      call X_train_min_nd%destroy
      call X_train_max_nd%destroy 
      call Y_train_min_d_nd%destroy 
	  call Y_train_max_d_nd%destroy      
      call python_nd%destroy
      call predict_results_std%destroy
      call predict_results_std_nd%destroy
     
      end subroutine      
         
     
         