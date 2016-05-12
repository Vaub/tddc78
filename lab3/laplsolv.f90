program laplsolv
!-----------------------------------------------------------------------
! Serial program for solving the heat conduction problem 
! on a square using the Jacobi method. 
! Written by Fredrik Berntsson (frber@math.liu.se) March 2003
! Modified by Berkant Savas (besav@math.liu.se) April 2006
!-----------------------------------------------------------------------
  integer, parameter                  :: n=100, maxiter=1000
  double precision,parameter          :: tol=1.0E-3
  double precision,dimension(0:n+1,0:n+1) :: T
  double precision,dimension(n)       :: tmp1,tmp2
  double precision,dimension(n+2)     :: tmp3
  double precision                    :: error,x,local_error
  real                                :: t1,t0
  integer                             :: i,j,k,c
  character(len=20)                   :: str
  logical                             :: is_set

  integer :: num_cpu, reminder, cpu, current_idx
  integer :: from_l, to_l, chunk_size  

  integer :: OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM

  ! Set boundary conditions and initial values for the unknowns
  T=0.0D0
  T(0:n+1 , 0)     = 1.0D0
  T(0:n+1 , n+1)   = 1.0D0
  T(n+1   , 0:n+1) = 2.0D0
  

  ! Solve the linear system of equations using the Jacobi method
  call cpu_time(t0)
  
  do k=1,maxiter     

     
     error=0.0D0
     current_idx = 1
     
     !$omp parallel             &
     !$omp default(private)     &
     !$omp shared(error, T, reminder, current_idx)
        
     num_cpu = OMP_GET_NUM_THREADS()
     cpu = OMP_GET_THREAD_NUM()
     chunk_size = n / num_cpu

     local_error = error 
     is_set = .FALSE.
     
     if (is_set .EQ. .FALSE.) then
     !$omp master
        reminder = modulo(n, num_cpu)
     !$omp end master
     !$omp barrier

     !$omp critical
        if (reminder > 0) then
            chunk_size = chunk_size + 1
            reminder = reminder - 1
        end if
        from_l = current_idx
        current_idx = from_l + chunk_size
        to_l = current_idx - 1
        
        !write(unit=*,fmt=*) 'Thread ',cpu,' from ',from_l,' to ',to_l
     !$omp end critical
     end if


     tmp1=T(from_l:to_l,0)
     do j=1,n        
        tmp2=T(from_l:to_l,j)
        tmp3=T(0:n+1,j)

        tmp3(from_l:to_l)=(tmp1+tmp3(from_l-1:to_l-1)+tmp3(from_l+1:to_l+1)+T(from_l:to_l,j+1))/4.0D0

        local_error=max(local_error,maxval(abs(tmp3(from_l:to_l)-tmp2)))
        tmp1=tmp2
        T(from_l:to_l,j)=tmp3(from_l:to_l)

        !$omp barrier
     end do

     !$omp critical
        error = max(local_error,error)
     !$omp end critical
     !$omp end parallel

     if (error<tol) then
        write(unit=*,fmt=*) 'Error too high!'
        exit
     end if
     
  end do
  
  call cpu_time(t1)

  write(unit=*,fmt=*) 'Time:',t1-t0,'Number of Iterations:',k
  write(unit=*,fmt=*) 'Temperature of element T(1,1)  =',T(1,1)

  ! Uncomment the next part if you want to write the whole solution
  ! to a file. Useful for plotting. 
  
  open(unit=7,action='write',file='result.dat',status='unknown')
  write(unit=str,fmt='(a,i6,a)') '(',N,'F10.6)'
  do i=0,n+1
     write (unit=7,fmt=str) T(i,0:n+1)  
  end do
  close(unit=7)
  
end program laplsolv
