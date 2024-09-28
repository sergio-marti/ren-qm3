!
! compile to a library (.so) by adding "-shared" on linux, or "-dynamiclib" on macOS
!
subroutine qm3_initialize
    use dynamo
    implicit none
    logical, dimension(:), allocatable :: flg

    open( unit=output, file="dynamo.log", status="replace", access="stream", form="formatted" )
    call dynamo_header

    call mm_file_process( "borra", "dynamo.opls" )
    call mm_system_construct( "borra", "dynamo.seq" )
    call xyz_read( "dynamo.xyz" )

    allocate( flg(1:natoms) )
    flg = atom_selection( subsystem = (/ "A" /) )
    call mopac_setup( method = "AM1", charge = 1, selection = flg )
    call energy_initialize
    call energy_non_bonding_options( &
        list_cutoff   = 12.8_dp, &
        outer_cutoff  = 12.0_dp, &
        inner_cutoff  = 10.0_dp, &
        minimum_image = .true. )
    deallocate( flg )
end subroutine qm3_initialize


subroutine qm3_update_coor( coor )
    use dynamo
    implicit none
    real*8, dimension(1:3,1:natoms), intent( in ) :: coor
    atmcrd = coor
end subroutine qm3_update_coor

subroutine qm3_update_chrg( chrg )
    use dynamo
    implicit none
    real*8, dimension(1:natoms), intent( in ) :: chrg
    atmchg = chrg
end subroutine qm3_update_chrg

subroutine qm3_get_func( coor, func )
    use dynamo
    implicit none
    real*8, dimension(1:3,1:natoms), intent( in ) :: coor
    real*8, intent( out ) :: func
    call qm3_update_coor( coor )
    call energy
    func = etotal
end subroutine qm3_get_func

subroutine qm3_get_grad( coor, func, grad )
    use dynamo
    implicit none
    real*8, dimension(1:3,1:natoms), intent( in ) :: coor
    real*8, intent( out ) :: func
    real*8, dimension(1:3,1:natoms), intent( out ) :: grad
    call qm3_update_coor( coor )
    call gradient
    func = etotal
    grad = atmder
end subroutine qm3_get_grad
