#ifndef TOP_ATTRS__TD
#define TOP_ATTRS__TD

include "TopDialect.td"
include "TopTraits.td"
include "TopTypes.td"

class Top_BaseAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Top_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def OpIndexAttr : Top_BaseAttr<"OpIndex", "OpIndex"> {
  let summary = "op index attribute";
  let description = [{
    op index description
  }];

  let parameters = (ins
      "OpIndexType":$op_index
  );

  let assemblyFormat = "`<` `id` `=` $op_index `>`";
}

def ShardingAttr : Top_BaseAttr<"sharding", "sharding", [ShardingTrait]> {
  let summary = "sharing attribute";
  let description = [{
    sharding description
  }];

  let parameters = (ins
      "int64_t":$num,
      ArrayRefParameter<"int64_t", "">:$shards
  );

  let genVerifyDecl = 1;

  let assemblyFormat = "`<` `shard_num` `=` $num `,` `shards` `=` `[` $shards `]` `>`";
}

#endif // TOP_ATTRS__TD