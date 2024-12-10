import sys
import os
import socket
import struct
import time
import typing


class client_fsi( object ):
    def __send( self, sckt, msg ):
        tmp = b"%010d"%( len( msg ) ) + msg
        siz = len( tmp )
        cur = 0
        while( cur < siz ):
            cur += sckt.send( tmp[cur:] )

    def __recv( self, sckt ):
        msg = sckt.recv( self.slen )
        siz = int( msg[0:10] )
        msg = msg[10:]
        cur = len( msg )
        while( cur < siz ):
            msg += sckt.recv( min( siz - cur, self.slen ) )
            cur = len( msg )
        return( msg )

    def __serve( self, chld ):
        msg = self.__recv( chld )
        cmd = chr( msg[0] )
        who = int( msg[1:4] )
        dst = int( msg[4:7] )
        if( cmd == "W" ):
            self.data[dst][who] += msg[7:]
        elif( cmd == "R" ):
            siz = int( msg[7:] )
            if( len( self.data[who][dst] ) >= siz ):
                self.__send( chld, self.data[who][dst][0:siz] )
                self.data[who][dst] = self.data[who][dst][siz:]
            else:
                self.__send( chld, b"" )
        elif( cmd == "X" ):
            self.sdwn -= 1
        elif( cmd == "B" ):
            self.barB += 1
        elif( cmd == "b" ):
            if( self.barB % self.ncpu == 0 ):
                chld.send( b"@#" )
            else:
                chld.send( b"@@" )
        elif( cmd == "P" ):
            self.barP += 1
        elif( cmd == "p" ):
            if( self.barP % self.ncpu == 0 ):
                chld.send( b"@#" )
            else:
                chld.send( b"@@" )
        chld.close()
        if( self.sdwn <= 0 ):
            os._exit( 0 )

    # on overloaded systems "wait" should be increased...
    def __init__( self, nproc, wait: typing.Optional[float] = 1.0 ):
        self.node = -1
        self.ncpu = nproc
        self.unix = "client_fsi.%d"%( os.getpid() )
        self.slen = 10240
        pids = list( range( self.ncpu ) )
        for i in range( self.ncpu ):
            if( self.node == -1 ):
                if( os.fork() == 0 ):
                    time.sleep( wait )
                    sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
                    sckt.connect( self.unix )
                    self.node = struct.unpack( "i", sckt.recv( 4 ) )[0]
                    sckt.close()
                    time.sleep( wait )
        if( self.node == -1 ):
            self.sdwn = self.ncpu
            self.barB = 0
            self.barP = 0
            self.data = [ [ b"" for j in range( self.ncpu ) ] for i in range( self.ncpu ) ]
            sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
            sckt.bind( self.unix )
            while( len( pids ) > 0 ):
                sckt.listen( self.ncpu * 2 )
                chld, addr = sckt.accept()
                chld.send( struct.pack( "i", pids.pop() ) )
                chld.close()
            sys.stderr.write( "fsi_server: %d processes initialized!\n"%( self.ncpu ) )
            while( True ):
                sckt.listen( self.ncpu * 2 )
                chld, addr = sckt.accept()
                self.__serve( chld )
            #sckt.close()

    def send_i4( self, dst, lst ):
        msg = b"W%03d%03d"%( self.node, dst )
        for r in lst:
            msg += struct.pack( "i", int( r ) )
        sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
        sckt.connect( self.unix )
        self.__send( sckt, msg )
        sckt.close()

    def recv_i4( self, src, siz ):
        sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
        sckt.connect( self.unix )
        self.__send( sckt, b"R%03d%03d%010d"%( self.node, src, siz * 4 ) )
        msg = self.__recv( sckt )
        sckt.close()
        while( len( msg ) < siz * 4 ):
            time.sleep( 0.1 )
            sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
            sckt.connect( self.unix )
            self.__send( sckt, b"R%03d%03d%010d"%( self.node, src, siz * 4 - len( msg ) ) )
            msg += self.__recv( sckt )
            sckt.close()
        return( list( struct.unpack( "%di"%( siz ), msg ) ) )

    def send_r8( self, dst, lst ):
        msg = b"W%03d%03d"%( self.node, dst )
        for r in lst:
            msg += struct.pack( "d", float( r ) )
        sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
        sckt.connect( self.unix )
        self.__send( sckt, msg )
        sckt.close()

    def recv_r8( self, src, siz ):
        sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
        sckt.connect( self.unix )
        self.__send( sckt, b"R%03d%03d%010d"%( self.node, src, siz * 8 ) )
        msg = self.__recv( sckt )
        sckt.close()
        while( len( msg ) < siz * 8 ):
            time.sleep( 0.01 )
            sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
            sckt.connect( self.unix )
            self.__send( sckt, b"R%03d%03d%010d"%( self.node, src, siz * 8 - len ( msg ) ) )
            msg += self.__recv( sckt )
            sckt.close()
        return( list( struct.unpack( "%dd"%( siz ), msg ) ) )

    def barrier( self ):
        sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
        sckt.connect( self.unix )
        self.__send( sckt, b"B%03d%03d_"%( self.node, 0 ) )
        sckt.close()
        flg = True
        while( flg ):
            time.sleep( 0.01 )
            sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
            sckt.connect( self.unix )
            self.__send( sckt, b"b%03d%03d_"%( self.node, 0 ) )
            flg = sckt.recv( 2 ) == b"@@"
            sckt.close()
        sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
        sckt.connect( self.unix )
        self.__send( sckt, b"P%03d%03d_"%( self.node, 0 ) )
        sckt.close()
        flg = True
        while( flg ):
            time.sleep( 0.01 )
            sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
            sckt.connect( self.unix )
            self.__send( sckt, b"p%03d%03d_"%( self.node, 0 ) )
            flg = sckt.recv( 2 ) == b"@@"
            sckt.close()

    def stop( self ):
        sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
        sckt.connect( self.unix )
        self.__send( sckt, b"X%03d%03d_"%( self.node, 0 ) )
        sckt.close()
        os._exit( 0 )



try:
    import  qm3.utils._mpi
    class client_mpi( object ):
        def __init__( self ):
            self.node, self.ncpu = qm3.utils._mpi.init()

        def barrier( self ):
            qm3.utils._mpi.barrier()

        def stop( self ):
            qm3.utils._mpi.stop()

        def send_r8( self, dst, lst ):
            qm3.utils._mpi.send_r8( dst, lst )

        def recv_r8( self, src, siz ):
            return( qm3.utils._mpi.recv_r8( src, siz ) )

        def send_i4( self, dst, lst ):
            qm3.utils._mpi.send_i4( dst, lst )

        def recv_i4( self, src, siz ):
            return( qm3.utils._mpi.recv_i4( src, siz ) )

except:
    pass
