# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""TVM Script Parser Typing Class for TIR

This module provides typing class for TVM script type annotation usage, it can be viewed as
a wrapper for uniform Type system in IR
"""
# pylint: disable=invalid-name
import tvm
from .special_stmt import SpecialStmt, convert_to_int


class TypeGeneric:  # pylint: disable=too-few-public-methods
    """Base class for all the TVM script typing class"""

    def evaluate(self):
        """Return an actual ir.Type Object that this Generic class wraps"""
        raise TypeError("Cannot get tvm.Type from a generic type")

    # This function is added here to avoid a pylint error
    # for T.int/float below not being callable
    def __call__(self):
        raise NotImplementedError()


class ConcreteType(TypeGeneric):  # pylint: disable=too-few-public-methods, abstract-method
    """TVM script typing class for uniform Type objects"""

    def __init__(self, vtype):
        self.type = vtype

    def evaluate(self):
        return tvm.ir.PrimType(self.type)


class GenericPtrType(TypeGeneric):  # pylint: disable=abstract-method
    """TVM script typing class generator for PtrType

    [] operator is overloaded, accepts a ConcreteType and returns a ConcreteType wrapping PtrType
    """

    def __getitem__(self, vtype):
        return ConcreteType(tvm.ir.PointerType(vtype.evaluate()))


class GenericTupleType(TypeGeneric):  # pylint: disable=abstract-method
    """TVM script typing class generator for TupleType

    [] operator is overloaded, accepts a list of ConcreteType and returns a ConcreteType
    wrapping TupleType
    """

    def __getitem__(self, vtypes):
        return ConcreteType(tvm.ir.TupleType([vtype.evaluate() for vtype in vtypes]))


class GenericBufferType(SpecialStmt):  # pylint: disable=too-few-public-methods, abstract-method
    """TVM script typing class for uniform Type objects"""

    def __init__(self, vtype):
        def match_buffer_syntax_sugar(
            shape,
            dtype: str = "float32",
            name: str = None,
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            span=None,
        ):
            if strides is None:
                strides = []
            align = convert_to_int(align, "align", self.context.report_error, self.node.span)
            offset_factor = convert_to_int(
                offset_factor, "offset_factor", self.context.report_error, self.node.span
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                span=span,
            )
            return buffer

        self.type = vtype
        super().__init__(match_buffer_syntax_sugar, def_symbol=True)

    def __call__(
        self,
        shape,
        dtype="float32",
        *,
        name: str = None,
        data=None,
        strides=None,
        elem_offset=None,
        scope="global",
        align=-1,
        offset_factor=0,
        buffer_type="default",
        span=None,
    ):
        """
        This function is for Buffer(...) syntax sugar.
        """
        pass  # pylint: disable=unnecessary-pass

    def __getitem__(self, args):
        """
        This function is for Buffer[...] syntax sugar
        Note that args is the list of all arguments
        """
        pass  # pylint: disable=unnecessary-pass


int8 = ConcreteType("int8")
int16 = ConcreteType("int16")
int32 = ConcreteType("int32")
int64 = ConcreteType("int64")
float16 = ConcreteType("float16")
float32 = ConcreteType("float32")
float64 = ConcreteType("float64")
boolean = ConcreteType("bool")
handle = ConcreteType("handle")
Ptr = GenericPtrType()
Tuple = GenericTupleType()
# we don't have 'buffer' type on the cpp side
# thus 'handle' is used here for convenience's sake
Buffer = GenericBufferType("handle")
