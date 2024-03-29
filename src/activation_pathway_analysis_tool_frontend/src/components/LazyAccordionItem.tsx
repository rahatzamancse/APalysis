import React from 'react'
import { Accordion } from 'react-bootstrap'

function LazyAccordionItem({ header, children, eventKey, className='', headerColor='black' }: { header: React.ReactNode, children: React.ReactNode, eventKey: string, className?: string, headerColor?: string }) {

  const [open, setOpen] = React.useState(false)

  return <>
    <Accordion.Item eventKey={eventKey} className={className}>
      <Accordion.Header className="accordion_header" onClick={() => {
        if(open)
          // TODO: Fix the glitchy behaviour of the accordion
          // setTimeout(() => setOpen(false), 1000)
          setOpen(false)
        else
          setOpen(true)
      }}>
        <span style={{color: headerColor}}>{header}</span>
      </Accordion.Header>
      <Accordion.Body>
        {open && children}
      </Accordion.Body>

    </Accordion.Item>
  </>
}

export default LazyAccordionItem