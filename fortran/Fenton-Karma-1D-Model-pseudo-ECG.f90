 program Fenton_Karma_1D_Model_pseudo_ECG_Simp
!*******************************************************************************
!                          <<< P U R P O S E >>>
! -->> Integrates in time the one-dimensional Fenton-Karma model 
! -->> 
! -->> Reference: 
!      Flavio Fenton, and Alain Karma
!      Chaos 8, 20 (1998); https://doi.org/10.1063/1.166311
!      ``Vortex dynamics in three-dimensional continuous myocardium with fiber 
!        rotation: Filament instability and fibrillation''.
! -->> 
!*******************************************************************************
! Nick Lazarides; last modified May 22, 2023./ 
!*******************************************************************************
 implicit none
 integer, parameter :: rk=selected_real_kind(15,307)
 integer, parameter :: nx=400
 integer, parameter :: neq=3*nx

 real(rk), parameter :: ell=3d0
 real(rk), parameter :: dx=ell/real(nx-1)
 real(rk), parameter :: twodx=2*dx

 integer, parameter :: np_trans=1
 integer, parameter :: np_stead=5

 integer, parameter :: io=10
 integer, parameter :: io2=20
 integer, parameter :: io3=50
 integer, parameter :: io4=100
 integer, parameter :: io5=200
 integer, parameter :: io6=500
 integer, parameter :: io7=1000
 integer, parameter :: io8=2000

 integer :: i, j, m, ii, iii

 real(rk), parameter :: pi=4d0*atan(1d0)
 real(rk), parameter :: twopi=2d0*pi

 real(rk), parameter :: period=600
 real(rk), parameter :: dt=1d-03
 real(rk), parameter :: nstep=nint(period/dt)

 integer, parameter :: p_wid=15
 real(rk), parameter :: j_amp=0.9d0
 real(rk), parameter :: p_dur=11d0

 real(rk), parameter :: tilde_d=0.005d0 ! in cm^2 / ms...

 real(rk), parameter :: c_m=1d0         ! in \mu F / cm^2...
 real(rk), parameter :: v_o=-85d0       ! in mV...
 real(rk), parameter :: v_fi=+15d0      ! in mV...
 real(rk), parameter :: k=10d0

 real(rk), parameter :: x_star=ell+2d0

 real(rk), parameter :: kappa=0.02d0

 real(rk), parameter :: hh=dt*0.5d0
 real(rk), parameter :: h6=dt/6d0


 real(rk), parameter, dimension(nx) :: x=(/ ( real(i), i=1,nx ) /)
 real(rk), parameter, dimension(nx) :: xx=(/ (real(i-1)*ell/(nx-1), i=1,nx) /)

 real(rk) :: t, tc, tc1, s 
 real(rk) :: bar_g_fi
 real(rk) :: tau_r, tau_si, tau_0
 real(rk) :: tau_v_plus, tau_v1_minus, tau_v2_minus
 real(rk) :: tau_w_plus, tau_w_minus
 real(rk) :: u_c, u_v, u_c_si
 real(rk) :: tau_d
 real(rk) :: ecg_star 

 real(rk), dimension(nx) :: vp
 real(rk), dimension(nx) :: dvdx, dgdx, g
 real(rk), dimension(nx) :: jst

 real(rk), dimension(neq) :: y, ydot


! open i/o units...
 open(08,file='fk-1d-cardyn-ecg-simp-x008')
 open(09,file='fk-1d-cardyn-ecg-simp-x009')
 open(10,file='fk-1d-cardyn-ecg-simp-x010')
 open(11,file='fk-1d-cardyn-ecg-simp-x011')
 open(12,file='fk-1d-cardyn-ecg-simp-x012')
 open(13,file='fk-1d-cardyn-ecg-simp-x013')
 open(14,file='fk-1d-cardyn-ecg-simp-x014')


 call load_parameter_set( )

 tau_d=c_m / bar_g_fi 

 call write_parameters_comments( )
 call load_write_initial_data( )

!-------------------------------------------------------------------------------
! time-integration...
!   #1 -> transient...
 ii=np_stead+1
   transient_periods_np_trans_iii_loop: do iii=1,np_trans
       single_period_nstep_j_loop: do j=1,nstep/2
         call derivs( t, y, ydot ); call rk4; t=period*real(iii-1) +dt*real(j)
         vp=(v_fi -v_o)*y(1:nx) +v_o
         ecg_star=0d0
         call calculate_pseudo_ecg( )
         call write_results( )

           if ( mod(j,io7) == 0 ) then
             write_periodically_uvw_profiles_loop: do m=1,nx
                 if ( abs( y(m+0*nx) ) < 1d-30 ) y(m+0*nx)=0d0
                 if ( abs( y(m+1*nx) ) < 1d-30 ) y(m+1*nx)=0d0
                 if ( abs( y(m+2*nx) ) < 1d-30 ) y(m+2*nx)=0d0
                 if ( abs( vp(m)  ) < 1d-30 ) vp(m)=0d0
               write(08,103) t, xx(m), vp(m)
             end do write_periodically_uvw_profiles_loop
           write(08,*)
         end if

       end do single_period_nstep_j_loop
   end do transient_periods_np_trans_iii_loop
 tc=t
!!! write(08,*) 'transient time: tc=', tc


!   #2 -> steady-state...
   steady_state_np_stead_ii_loop: do ii=1,np_stead

       single_period_nstep_j_loop_2: do j=1,nstep
         call derivs( t, y, ydot ); call rk4; t=tc+period*real(ii-1) +dt*real(j)
         s=t-tc

         vp=(v_fi -v_o)*y(1:nx) +v_o

         ecg_star=0d0
         call calculate_pseudo_ecg( )
         call write_results( )

           if ( mod(j,io7) == 0 ) then
             write_periodically_uvw_profiles_ss_loop: do m=1,nx
                 if ( abs( y(m+0*nx) ) < 1d-30 ) y(m+0*nx)=0d0
                 if ( abs( y(m+1*nx) ) < 1d-30 ) y(m+1*nx)=0d0
                 if ( abs( y(m+2*nx) ) < 1d-30 ) y(m+2*nx)=0d0
                 if ( abs( vp(m)  ) < 1d-30 ) vp(m)=0d0
!!!               write(08,106) t, xx(m), vp(m), y(m), y(m+nx), y(m+2*nx)
               write(08,103) t, xx(m), vp(m)
             end do write_periodically_uvw_profiles_ss_loop
           write(08,*)
         end if

         if ( t > 600d0 ) stop

       end do single_period_nstep_j_loop_2
   end do steady_state_np_stead_ii_loop
! end time-integration...
!-------------------------------------------------------------------------------

! formats...
 102 format(2(1pe12.4)); 202 format(2(1pe14.6)); 302 format(2(1pe16.8))
 103 format(3(1pe12.4)); 203 format(3(1pe14.6)); 303 format(3(1pe16.8))
 104 format(4(1pe12.4)); 204 format(4(1pe14.6)); 304 format(4(1pe16.8))
 105 format(5(1pe12.4)); 205 format(5(1pe14.6)); 305 format(5(1pe16.8))
 106 format(6(1pe12.4)); 206 format(6(1pe14.6)); 306 format(6(1pe16.8))


!********************************************************************************
 contains
!********************************************************************************


!*******************************************************************************
 subroutine write_results( )
 implicit none
 integer, parameter :: rk=selected_real_kind(15,307)
 integer :: tn

   if ( mod(j,io6) == 0 ) then
     write(09,306) t, vp(16),   vp(32),   vp(64),   vp(nx/2),   vp(nx)
     write(10,306) t, y(16),    y(32),    y(64),    y(nx/2),    y(nx)
     write(11,306) t, y(nx+16), y(nx+32), y(nx+64), y(nx+nx/2), y(nx+nx)
     tn=2*nx
     write(12,306) t, y(tn+16), y(tn+32), y(tn+64), y(tn+nx/2), y(tn+nx)

     write(13,302) t, ecg_star 
   end if

 302 format(2(1pe16.8)); 306 format(6(1pe16.8))
 end subroutine write_results
!*******************************************************************************


!*******************************************************************************
 subroutine load_write_initial_data( )
 implicit none
 integer, parameter :: rk=selected_real_kind(15,307)

 t=0d0; y=0d0

 vp=(v_fi -v_o)*y(1:nx) +v_o

   write_initial_uvw_profiles_loop: do j=1,nx
       if ( abs( y(j+0*nx) ) < 1d-30 ) y(j+0*nx)=0d0
       if ( abs( y(j+1*nx) ) < 1d-30 ) y(j+1*nx)=0d0
       if ( abs( y(j+2*nx) ) < 1d-30 ) y(j+2*nx)=0d0
       if ( abs( vp(j)  ) < 1d-30 ) vp(j)=0d0
     write(08,106) 0d0, xx(j), vp(j), y(j+0*nx), y(j+1*nx), y(j+2*nx)
   end do write_initial_uvw_profiles_loop
 write(08,*)

 106 format(6(1pe12.4))

 end subroutine load_write_initial_data
!*******************************************************************************


!*******************************************************************************
 subroutine calculate_pseudo_ecg( )
 implicit none
 integer, parameter :: rk=selected_real_kind(15,307)

 g=1d0/abs(x_star -xx)

 dgdx(1)=0d0
 dvdx(1)=0d0
   do i=2,nx-1
     dgdx(i)=( g(i+1) -g(i-1) ) / twodx
     dvdx(i)=( vp(i+1) -vp(i-1) ) / twodx
   end do
 dgdx(nx)=0d0
 dvdx(nx)=0d0

 ecg_star=-kappa*sum( dvdx*dgdx )*dx

 end subroutine calculate_pseudo_ecg
!*******************************************************************************


!*******************************************************************************
 subroutine load_parameter_set( )
 implicit none
 integer, parameter :: rk=selected_real_kind(15,307)

! modified Beeler-Reuter (MBR) model... 
 bar_g_fi    =4d0
 tau_r       =50d0
 tau_si      =44.84d0
 tau_0       =8.3d0
 tau_v_plus  =3.33d0
 tau_v1_minus=1000d0
 tau_v2_minus=19.2d0
 tau_w_plus  =667d0
 tau_w_minus =11d0
 u_c         =0.13d0
 u_v         =0.055d0
 u_c_si      =0.85d0

 end subroutine load_parameter_set 
!*******************************************************************************


!*******************************************************************************
 subroutine rk4
 implicit none
 integer, parameter :: rk=selected_real_kind(15,307)
 real(rk) :: xh, tpdt
 real(rk), dimension(neq) :: yt, dyt, dym

 xh=t+hh; tpdt=t+dt

 yt=y+hh*ydot; call derivs(xh,yt,dyt)
 yt=y+hh*dyt; call derivs(xh,yt,dym)
 yt=y+dt*dym; dym=dyt+dym; call derivs(tpdt,yt,dyt)
 y=y+h6*(ydot+dyt+2d0*dym)

 end subroutine rk4
!*******************************************************************************


!*******************************************************************************
 subroutine derivs( t, y, ydot )
 implicit none
 integer, parameter :: rk=selected_real_kind(15,307)
 integer :: i

 real(rk) :: t

 real(rk), dimension(nx) :: jfi, jso, jsi, jion
 real(rk), dimension(nx) :: d2udx2 
 real(rk), dimension(nx) :: tau_v_minus

 real(rk), dimension(neq) :: y, ydot

 call stimulus_current( )

   do i=1,nx
     if ( y(i) >= u_v ) then
       tau_v_minus(i)=+tau_v1_minus 
     else if ( y(i) < u_v ) then
       tau_v_minus(i)=+tau_v2_minus
     end if
   end do

! Neumann Boundary Conditions...
 d2udx2(1)=+2d0*( y(2) -y(1) ) / dx**2
   do i=2,nx-1
     d2udx2(i)=( y(i+1) -2d0*y(i) +y(i-1) ) / dx**2
   end do
 d2udx2(nx)=-2d0*( y(nx) -y(nx-1) ) / dx**2

! equations...
   do i=1,nx
     if ( y(i) >= u_c ) then

!   ionic currents...
       jfi(i)=-y(nx+i)*( 1d0 -y(i) )*( y(i) -u_c ) / tau_d
       jso(i)=+1d0 / tau_r
       jsi(i)=-y(2*nx+i)*( 1d0 +tanh( k*(y(i) -u_c_si) ) ) / (2d0*tau_si)

       jion(i)=jfi(i) +jso(i) +jsi(i)

!   set of equations...
       ydot(i)=+tilde_d*d2udx2(i) -jion(i) +jst(i)
       ydot(nx+i)=-y(nx+i) / tau_v_plus
       ydot(2*nx+i)=-y(2*nx+i) / tau_w_plus

     else if ( y(i) < u_c ) then

!   ionic currents...
       jfi(i)=0d0
       jso(i)=+y(i) / tau_0
       jsi(i)=-y(2*nx+i)*( 1d0 +tanh(k*(y(i) -u_c_si)) ) / (2d0*tau_si)

       jion(i)=jfi(i) +jso(i) +jsi(i)

!   equations...
       ydot(i)=+tilde_d*d2udx2(i) -jion(i) +jst(i)
       ydot(nx+i)=( 1d0 -y(nx+i) ) / tau_v_minus(i)
       ydot(2*nx+i)=( 1d0 -y(2*nx+i) ) / tau_w_minus

     end if
   end do

 end subroutine derivs
!*******************************************************************************


!*******************************************************************************
 subroutine stimulus_current( )
 implicit none
 integer, parameter :: rk=selected_real_kind(15,307)
 integer :: q

 jst=0d0
   if ( s > period*real(ii-1) .and. s < period*real(ii-1)+p_dur ) then
     jst(1:p_wid)=j_amp
   else
     jst(1:p_wid)=0d0
   end if

   if ( mod(j,io6) == 0 ) write(14,102) t, jst(1) 
!   if ( mod(j,io8) == 0 ) then
!     write(14,103) ( t, xx(q), jst(q), q=1,nx ) 
!     write(14,*) 
!   end if

 102 format(2(1pe12.4)); 103 format(3(1pe12.4))

 end subroutine stimulus_current
!*******************************************************************************


!*******************************************************************************
 subroutine write_parameters_comments( )
 implicit none
 integer, parameter :: rk=selected_real_kind(15,307)

 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## Program:     Fenton-Karma-1D-Model-pseudo-ECG-Simp.f90        '
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '##                     P A R A M E T E R S                       '
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## nx =', nx
 write(08,*) '## neq=', neq
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## np_trans=', np_trans
 write(08,*) '## np_stead=', np_stead
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## dt    =', dt
 write(08,*) '## period=', period
 write(08,*) '## nstep =', nstep
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## dx =', dx
 write(08,*) '## ell=', ell
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## p_wid=', p_wid
 write(08,*) '## j_amp=', j_amp
 write(08,*) '## p_dur=', p_dur
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## tilde_d=', tilde_d 
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## c_m =', c_m
 write(08,*) '## v_o =', v_o
 write(08,*) '## v_fi=', v_fi
 write(08,*) '## k   =', k
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## x_star=', x_star 
 write(08,*) '## kappa =', kappa
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## Modified Beeler Reuter Model./                                '
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## bar_g_fi    =', bar_g_fi 
 write(08,*) '## tau_r       =', tau_r
 write(08,*) '## tau_si      =', tau_si
 write(08,*) '## tau_0       =', tau_0
 write(08,*) '## tau_v_plus  =', tau_v_plus
 write(08,*) '## tau_v1_minus=', tau_v1_minus
 write(08,*) '## tau_v2_minus=', tau_v2_minus
 write(08,*) '## tau_w_plus  =', tau_w_plus
 write(08,*) '## tau_w_minus =', tau_w_minus
 write(08,*) '## u_c         =', u_c
 write(08,*) '## u_v         =', u_v
 write(08,*) '## u_c_si      =', u_c_si
 write(08,*) '## tau_d       =', tau_d
 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## --------------------------------------------------------------'


 write(08,*) '## --------------------------------------------------------------'
 write(08,*) '## '
 write(08,*) '## --------------------------------------------------------------'

 write(09,*) '## --------------------------------------------------------------'
 write(09,*) '## '
 write(09,*) '## --------------------------------------------------------------'

 write(10,*) '## --------------------------------------------------------------'
 write(10,*) '## '
 write(10,*) '## --------------------------------------------------------------'

 write(11,*) '## --------------------------------------------------------------'
 write(11,*) '## '
 write(11,*) '## --------------------------------------------------------------'

 write(12,*) '## --------------------------------------------------------------'
 write(12,*) '## '
 write(12,*) '## --------------------------------------------------------------'

 end subroutine write_parameters_comments
!*******************************************************************************


!*******************************************************************************
 end program Fenton_Karma_1D_Model_pseudo_ECG_Simp
!*******************************************************************************
