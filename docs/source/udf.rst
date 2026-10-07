.. _api.udf:

===========================
User Defined Functions
===========================

______

AutoDataPrep
----------------

.. currentmodule:: verticapy

.. autofunction:: sdk.vertica.udf.gen.generate_lib_udf

Python Scalar UDF Registration
------------------------------

The high-level API registers a Python scalar function through Vertica's
existing Python UDx library mechanism and returns an object that can be used
to build VerticaPy SQL expressions.

.. code-block:: python

	from verticapy import udf

	@udf(name="add_one")
	def add_one(value: int) -> int:
		return value + 1

	expression = add_one(vdf["value"])

The explicit registration form is useful when a connection should be selected
for a group of registrations:

.. code-block:: python

	from verticapy.udf import UDFRegistration

	registration = UDFRegistration()
	add_one = registration.register(
		lambda value: value + 1,
		input_types=[int],
		return_type=int,
		name="add_one",
	)

Type annotations can replace ``input_types`` and ``return_type``. Supported
Python scalar types are ``bool``, ``int``, ``float``, ``str``, and ``bytes``.
The generated library file must be accessible to the Vertica server, as with
the lower-level UDx APIs.