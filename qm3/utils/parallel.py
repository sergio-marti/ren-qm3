import sys
import os
import socket
import struct
import time
import typing


class client_fsi( object ):
    def __send( self, sckt, msg ):
        tmp = struct.pack( "i", len( msg ) ) + msg
        sckt.sendall( tmp )

    def __recv( self, sckt ):
        msg = sckt.recv( self.slen )
        siz = struct.unpack( "i", msg[0:4] )[0]
        msg = msg[4:]
        cur = len( msg )
        while( cur < siz ):
            msg += sckt.recv( min( siz - cur, self.slen ) )
            cur = len( msg )
        return( msg )

    def __serve( self, chld ):
        msg = self.__recv( chld )
        cmd = chr( msg[0] )
        who = struct.unpack( "h", msg[1:3] )[0]
        dst = struct.unpack( "h", msg[3:5] )[0]
        if( cmd == "W" ):
            self.data[dst][who].extend( msg[5:] )
        elif( cmd == "R" ):
            siz = struct.unpack( "i", msg[5:] )[0]
            #if( len( self.data[who][dst] ) >= siz ):
            #    self.__send( chld, self.data[who][dst][0:siz] )
            #    del self.data[who][dst][0:siz]
            #else:
            #    self.__send( chld, b"" )
            self.__send( chld, self.data[who][dst][0:siz] )
            del self.data[who][dst][0:siz]
        elif( cmd == "B" ):
            self.barB += 1
        elif( cmd == "b" ):
            chld.send( b"@@" if( self.barB % self.ncpu == 0 ) else b"__" )
        elif( cmd == "P" ):
            self.barP += 1
        elif( cmd == "p" ):
            chld.send( b"@@" if( self.barP % self.ncpu == 0 ) else b"__" )
        elif( cmd == "X" ):
            self.sdwn -= 1

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
                    with socket.socket( socket.AF_UNIX, socket.SOCK_STREAM ) as sckt:
                        sckt.connect( self.unix )
                        self.__node = sckt.recv( 2 )
                        self.node = struct.unpack( "h", self.__node )[0]
                    time.sleep( wait )

        if( self.node == -1 ):
            self.sdwn = self.ncpu
            self.barB = 0
            self.barP = 0
            self.data = [ [ bytearray() for _ in range( self.ncpu ) ] for _ in range( self.ncpu ) ]
            sckt = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
            sckt.bind( self.unix )

            while( len( pids ) > 0 ):
                sckt.listen( self.ncpu * 2 )
                chld, addr = sckt.accept()
                chld.send( struct.pack( "h", pids.pop() ) )
                chld.close()
            sys.stderr.write( "fsi_server: %d processes initialized!\n"%( self.ncpu ) )

            while( True ):
                sckt.listen( self.ncpu * 2 )
                chld, addr = sckt.accept()
                self.__serve( chld )
            #sckt.close()

    def send_i4( self, dst, lst ):
        msg = b"W" + self.__node + struct.pack( "h", dst ) + b"".join( struct.pack( "i", x ) for x in lst )
        with socket.socket( socket.AF_UNIX, socket.SOCK_STREAM ) as sckt:
            sckt.connect( self.unix )
            self.__send( sckt, msg )

    def send_r8( self, dst, lst ):
        msg = b"W" + self.__node + struct.pack( "h", dst ) + b"".join( struct.pack( "d", x ) for x in lst )
        with socket.socket( socket.AF_UNIX, socket.SOCK_STREAM ) as sckt:
            sckt.connect( self.unix )
            self.__send( sckt, msg )

    def __xrecv( self, src, siz, knd, fmt ):
        dim = siz * knd
        msg = bytearray()
        with socket.socket( socket.AF_UNIX, socket.SOCK_STREAM ) as sckt:
            sckt.connect( self.unix )
            self.__send( sckt, b"R" + self.__node + struct.pack( "h", src ) + struct.pack( "i", dim ) )
            msg.extend( self.__recv( sckt ) )
        while( len( msg ) < dim ):
            time.sleep( 0.01 )
            with socket.socket( socket.AF_UNIX, socket.SOCK_STREAM ) as sckt:
                sckt.connect( self.unix )
                self.__send( sckt, b"R" + self.__node + struct.pack( "h", src ) + struct.pack( "i", dim - len( msg ) ) )
                msg.extend( self.__recv( sckt ) )
        return( list( struct.unpack( "%d%s"%( siz, fmt ), msg ) ) )

    def recv_i4( self, src, siz ):
        return( self.__xrecv( src, siz, 4, "i" ) )

    def recv_r8( self, src, siz ):
        return( self.__xrecv( src, siz, 8, "d" ) )

    def __xbarrier( self, sgnl, poll ):
        with socket.socket( socket.AF_UNIX, socket.SOCK_STREAM ) as sckt:
            sckt.connect( self.unix )
            self.__send( sckt, sgnl + self.__node + b"\x00\x00_" )
        while( True ):
            time.sleep( 0.01 )
            with socket.socket( socket.AF_UNIX, socket.SOCK_STREAM ) as sckt:
                sckt.connect( self.unix )
                self.__send( sckt, poll + self.__node + b"\x00\x00_" )
                if( sckt.recv( 2 ) == b"@@" ):
                    break

    def barrier( self ):
        self.__xbarrier( b"B", b"b" )
        self.__xbarrier( b"P", b"p" )

    def stop( self ):
        with socket.socket( socket.AF_UNIX, socket.SOCK_STREAM ) as sckt:
            sckt.connect( self.unix )
            self.__send( sckt, b"X" + self.__node + b"\x00\x00_" )
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
