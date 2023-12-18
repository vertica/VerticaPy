.. _api.connect:

===========
Connection
===========

______

Functions
--------------------

.. important::

   For a comprehensive guide to create connections, please refer to :ref:`connection`.

Read
~~~~~

.. currentmodule:: verticapy.connection

.. autosummary:: 
   :toctree: api/

   auto_connect
   available_connections
   connect
   current_connection
   current_cursor
   get_connection_file
   get_confparser
   read_dsn
   vertica_connection
   verticapylab_connection
   .. read_auto_connect


Write
~~~~~~

.. currentmodule:: verticapy.connection

.. autosummary:: 
   :toctree: api/

   change_auto_connection
   new_connection
   set_connection
   set_external_connection


Close/Delete
~~~~~~~~~~~~~~

.. currentmodule:: verticapy.connection

.. autosummary:: 
   :toctree: api/

   close_connection
   delete_connection

______

Global Connection
--------------------


.. currentmodule:: verticapy.connection

.. autosummary:: 
   :toctree: api/


   global_connection.GlobalConnection

**Methods:**

.. currentmodule:: verticapy.connection.global_connection

.. autosummary:: 
   :toctree: api/

   GlobalConnection.get_connection
   GlobalConnection.get_dsn
   GlobalConnection.get_dsn_section
   GlobalConnection.get_external_connections
   GlobalConnection.set_connection
   GlobalConnection.set_external_connections