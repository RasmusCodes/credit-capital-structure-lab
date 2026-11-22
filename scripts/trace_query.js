// Run in the browser console on https://www.finra.org/finra-data/fixed-income/corp-and-agency
// Returns TRACE last-sale data for the issuers below as a JSON string (save it as finra_last_trades.json).
(async () => {
  const tok = (document.cookie.match(/XSRF-TOKEN=([^;]+)/) || [])[1];
  const url = "https://services-dynarep.ddwa.finra.org/public/reporting/v2/data/group/FixedIncomeMarket/name/CorporateAndAgencySecurities";
  const fields = ["issueSymbolIdentifier", "issuerName", "couponRate", "maturityDate", "lastSalePrice", "lastSaleYield",
    "lastTradeDate", "cusip", "productSubTypeCode", "moodysRating", "standardAndPoorsRating", "isCallable", "traceGradeCode"];
  const out = {};
  for (const name of ["COREWEAVE", "NEBIUS", "IREN", "MICROSTRATEGY", "WOLFSPEED"]) {
    const body = {fields, compareFilters: [{fieldName: "issuerName", fieldValue: name, compareType: "BEGINS_WITH"}], limit: 200, offset: 0};
    const r = await fetch(url, {method: "POST", credentials: "include",
      headers: {"Content-Type": "application/json", "Accept": "application/json", "X-XSRF-TOKEN": tok}, body: JSON.stringify(body)});
    const j = await r.json();
    const rows = JSON.parse(j.returnBody.data);
    out[name] = rows.map(x => [x.issuerName, x.cusip, x.issueSymbolIdentifier, +x.couponRate, x.maturityDate,
      x.lastSalePrice === null ? null : +x.lastSalePrice, x.lastSaleYield === null ? null : +x.lastSaleYield,
      x.lastTradeDate, x.productSubTypeCode, x.moodysRating, x.standardAndPoorsRating, x.isCallable, x.traceGradeCode]);
  }
  return JSON.stringify({columns: ["issuerName", "cusip", "symbol", "coupon", "maturity", "lastSalePrice", "lastSaleYield",
    "lastTradeDate", "productSubType", "moodys", "sp", "callable", "traceGrade"], data: out});
})();
