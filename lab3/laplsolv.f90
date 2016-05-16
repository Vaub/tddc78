program laplsolv

  use omp_lib
  implicit none
!-----------------------------------------------------------------------
! Serial program for solving the heat conduction problem 
! on a square using the Jacobi method. 
! Written by Fredrik Berntsson (frber@math.liu.se) March 2003
! Modified by Berkant Savas (besav@math.liu.se) April 2006
!-----------------------------------------------------------------------
  integer, parameter                  :: n=250, maxiter=1000
  double precision,parameter          :: tol=1.0E-3
  double precision,dimension(0:n+1,0:n+1) :: T
  double precision,dimension(n)       :: tmp1,tmp2,tmp3,left_col,right_col
  double precision                    :: error,local_error,x
  double precision                    :: t1,t0
  integer                             :: i,j,k,cpu,nb_cpu_max
  character(len=20)                   :: str
  integer, dimension(:), allocatable  :: from_col(:), to_col(:)
  integer                             :: reminder, current_col, chunk_size

  
  
  ! Set boundary conditions and initial values for the unknowns
  T=0.0D0
  T(0:n+1 , 0)     = 1.0D0
  T(0:n+1 , n+1)   = 1.0D0
  T(n+1   , 0:n+1) = 2.0D0
  
  nb_cpu_max = omp_get_max_threads()

  allocate(from_col(1:nb_cpu_max))
  allocate(to_col(1:nb_cpu_max))
    
  current_col = 1
  chunk_size = n / nb_cpu_max
  reminder = modulo(n,nb_cpu_max)
  
  do cpu=1,nb_cpu_max
        from_col(cpu) = current_col
        to_col(cpu) = current_col + chunk_size - 1
        if (reminder > 0) then
            to_col(cpu) = to_col(cpu) + 1
            reminder = reminder - 1
        end if
        current_col = to_col(cpu) + 1
  end do

  ! Solve the linear system of equations using the Jacobi method
  t0 = omp_get_wtime()
  
  do k=1,maxiter
     error=0.0D0
     
     !$omp parallel             &
     !$omp default(private)     &
     !$omp shared(T,error,from_col,to_col)
     
     local_error = 0.0D0

     cpu = omp_get_thread_num() + 1
     left_col = T(1:n,from_col(cpu)-1)
     right_col = T(1:n,to_col(cpu)+1)

     !$omp barrier

     tmp1=left_col
     do j=from_col(cpu),to_col(cpu)
        tmp2=T(1:n,j)

        if (j+1 > to_col(cpu)) then
            tmp3 = right_col
        else
            tmp3 = T(1:n,j+1)
        endif

        T(1:n,j)=(T(0:n-1,j)+T(2:n+1,j)+tmp3+tmp1)/4.0D0
        tmp1=tmp2

        local_error = max(local_error, maxval(abs(tmp2-T(1:n,j))))
     end do

     !$omp atomic
     error=max(local_error,error)

     !$omp end parallel
     
     if (error<tol) then
        exit
     end if
     
  end do
  
  t1 = omp_get_wtime()

  write(unit=*,fmt=*) 'Time:',t1-t0,'Number of Iterations:',k
  write(unit=*,fmt=*) 'Temperature of element T(1,1)  =',T(1,1)

  ! Uncomment the next part if you want to write the whole solution
  ! to a file. Useful for plotting. 
  
  !open(unit=7,action='write',file='result.dat',status='unknown')
  !write(unit=str,fmt='(a,i6,a)') '(',N,'F10.6)'
  !do i=0,n+1
  !   write (unit=7,fmt=str) T(i,0:n+1)  
  !end do
  !close(unit=7)
  
end program laplsolv
