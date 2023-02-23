import React from 'react'
import { Accordion } from 'react-bootstrap'

function LazyAccordionItem({ header, children, eventKey }: { header: React.ReactNode, children: React.ReactNode, eventKey: string }) {

  const [open, setOpen] = React.useState(false)

  return <>
    <Accordion.Item eventKey={eventKey}>
      <Accordion.Header className="accordion_header" onClick={() => {
        if(open)
          // TODO: Fix the glitchy behaviour of the accordion
          // setTimeout(() => setOpen(false), 1000)
          setOpen(false)
        else
          setOpen(true)
      }}>{header}</Accordion.Header>
      <Accordion.Body>
        {open && children}
      </Accordion.Body>

    </Accordion.Item>
  </>
}

export default LazyAccordionItem