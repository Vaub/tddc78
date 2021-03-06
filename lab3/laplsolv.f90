program laplsolv

  use omp_lib
  implicit none
!-----------------------------------------------------------------------
! Serial program for solving the heat conduction problem 
! on a square using the Jacobi method. 
! Written by Fredrik Berntsson (frber@math.liu.se) March 2003
! Modified by Berkant Savas (besav@math.liu.se) April 2006
!-----------------------------------------------------------------------
  integer, parameter                  :: n=2000, maxiter=1000
  double precision,parameter          :: tol=1.0E-3
  double precision,dimension(0:n+1,0:n+1) :: T
  double precision,dimension(n)       :: tmp1,tmp2,tmp3,left_col,right_col
  double precision                    :: error,x
  double precision                    :: t1,t0
  integer                             :: i,j,k,cpu,nb_cpu
  integer*8                           :: flops
  character(len=20)                   :: str
  integer, dimension(:), allocatable  :: from_col(:), to_col(:)
  integer                             :: reminder, current_col, chunk_size

  
  
  ! Set boundary conditions and initial values for the unknowns
  T=0.0D0
  T(0:n+1 , 0)     = 1.0D0
  T(0:n+1 , n+1)   = 1.0D0
  T(n+1   , 0:n+1) = 2.0D0
 
  ! finding the num of threads 
  !$omp parallel shared(nb_cpu)
  nb_cpu = omp_get_max_threads()
  !$omp end parallel

  allocate(from_col(1:nb_cpu))
  allocate(to_col(1:nb_cpu))
    
  current_col = 1
  chunk_size = n / nb_cpu
  reminder = modulo(n,nb_cpu)
  
  ! every cpu gets a left/right column index
  do cpu=1,nb_cpu
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
  
  !flops = 0
  do k=1,maxiter
     error=0.0D0
     
     !$omp parallel reduction(MAX:error) &
     !$omp default(private)              &
     !$omp shared(T,from_col,to_col) 
     
     ! every threads get its "shared" column in local memory 
     cpu = omp_get_thread_num() + 1
     left_col = T(1:n,from_col(cpu)-1)
     right_col = T(1:n,to_col(cpu)+1)

     !$omp barrier

     tmp1=left_col
     do j=from_col(cpu),to_col(cpu)
        tmp2=T(1:n,j)

        ! if outside of thread scope, use its shared column
        if (j+1 > to_col(cpu)) then
            tmp3 = right_col
        else
            tmp3 = T(1:n,j+1)
        endif

        T(1:n,j)=(T(0:n-1,j)+T(2:n+1,j)+tmp3+tmp1)/4.0D0
        !flops = flops + n*4

        tmp1=tmp2

        error = max(error, maxval(abs(tmp2-T(1:n,j)))) ! will be reduced (MAX) by omp
        !flops = flops + (n * 2 + n + 1)
     end do

     !$omp end parallel
     
     if (error<tol) then
        exit
     end if
     
  end do
  
  t1 = omp_get_wtime()

  write(unit=*,fmt=*) 'Time:',t1-t0,'Number of Iterations:',k
  write(unit=*,fmt=*) 'Temperature of element T(1,1)  =',T(1,1)
  !write(unit=*,fmt=*) 'FLOP:',flops

  ! Uncomment the next part if you want to write the whole solution
  ! to a file. Useful for plotting. 
  
  open(unit=7,action='write',file='result.dat',status='unknown')
  write(unit=str,fmt='(a,i6,a)') '(',N,'F10.6)'
  do i=0,n+1
     write (unit=7,fmt=str) T(i,0:n+1)  
  end do
  close(unit=7)
  
end program laplsolv
