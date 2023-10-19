# Xpaths
XPATHS = {
    "download_pdf": "//*[local-name() = 'svg'][@data-icon = 'download']/*[local-name() = 'path']",
    "Fejlkode 404": "//h1[contains(text(), 'Fejlkode 404')]",
    "Accept cookies": "//a[@id='CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll']",
    "Øvrige sagsoplysninger": "//span[@class='accordion-title'][contains(text(), 'Øvrige sagsoplysninger')]",
}
XPATHS_TABULAR_DATA = {
    "Overskrift": "//h4[contains(text(), 'Overskrift')]/following-sibling::span[1]/p",
    "Afgørelsesstatus": "//h4[contains(text(), 'Afgørelsesstatus')]/following-sibling::p[1]",
    "Faggruppe": "//h4[contains(text(), 'Faggruppe')]/following-sibling::p[1]",
    "Ret": "//h4[contains(text(), 'Ret')]/following-sibling::p[1]",
    "Rettens sagsnummer": "//h4[contains(text(), 'Rettens sagsnummer')]/following-sibling::p[1]",
    "Sagstype": "//h4[contains(text(), 'Sagstype')]/following-sibling::p[1]",
    "Instans": "//h4[contains(text(), 'Instans')]/following-sibling::p[1]",
    "Domsdatabasens sagsnummer": "//h4[contains(text(), 'Domsdatabasens sagsnummer')]/following-sibling::p[1]",
    "Sagsemner": "//h4[contains(text(), 'Sagsemner')]/following-sibling::p[1]/span[1]",
    "Særlige retsskridt": "//h4[contains(text(), 'Særlige retsskridt')]/following-sibling::p[1]/span[1]",
    "Sagsdeltagere": "//h4[contains(text(), 'Sagsdeltagere')]/following-sibling::p[1]/span[1]",
    "Dørlukning": "//h4[contains(text(), 'Dørlukning')]/following-sibling::p[1]",
    "Løftet ud af småsagsprocessen": "//h4[contains(text(), 'Løftet ud af småsagsprocessen')]/following-sibling::p[1]",
    "Anerkendelsespåstand": "//h4[contains(text(), 'Anerkendelsespåstand')]/following-sibling::p[1]",
    "Politiets journalnummer": "//h4[contains(text(), 'Politiets journalnummer')]/following-sibling::p[1]",
    "Påstandsbeløb": "//h4[contains(text(), 'Påstandsbeløb')]/following-sibling::p[1]",
    "Sagskomplekser": "//h4[contains(text(), 'Sagskomplekser')]/following-sibling::p[1]/span[1]",
}
